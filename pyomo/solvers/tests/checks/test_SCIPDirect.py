#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys

import pyomo.common.unittest as unittest

from pyomo.environ import (
    ConcreteModel,
    AbstractModel,
    Var,
    Objective,
    Block,
    Constraint,
    Suffix,
    NonNegativeIntegers,
    NonNegativeReals,
    Integers,
    Binary,
    value,
)
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus

try:
    import pyscipopt

    scip_available = True
except ImportError:
    scip_available = False


@unittest.skipIf(not scip_available, "The SCIP python bindings are not available")
class SCIPDirectTests(unittest.TestCase):
    def setUp(self):
        self.stderr = sys.stderr
        sys.stderr = None

    def tearDown(self):
        sys.stderr = self.stderr

    def test_infeasible_lp(self):
        with SolverFactory("scip_direct", solver_io="python") as opt:
            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.C1 = Constraint(expr=model.X == 1)
            model.C2 = Constraint(expr=model.X == 2)
            model.O = Objective(expr=model.X)

            results = opt.solve(model)

            self.assertEqual(
                results.solver.termination_condition, TerminationCondition.infeasible
            )

    def test_unbounded_lp(self):
        with SolverFactory("scip_direct", solver_io="python") as opt:
            model = ConcreteModel()
            model.X = Var()
            model.O = Objective(expr=model.X)

            results = opt.solve(model)

            self.assertIn(
                results.solver.termination_condition,
                (
                    TerminationCondition.unbounded,
                    TerminationCondition.infeasibleOrUnbounded,
                ),
            )

    def test_optimal_lp(self):
        with SolverFactory("scip_direct", solver_io="python") as opt:
            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.O = Objective(expr=model.X)

            results = opt.solve(model, load_solutions=False)

            self.assertEqual(results.solution.status, SolutionStatus.optimal)

    def test_infeasible_mip(self):
        with SolverFactory("scip_direct", solver_io="python") as opt:
            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.C1 = Constraint(expr=model.X == 1)
            model.C2 = Constraint(expr=model.X == 2)
            model.O = Objective(expr=model.X)

            results = opt.solve(model)

            self.assertEqual(
                results.solver.termination_condition, TerminationCondition.infeasible
            )

    def test_unbounded_mip(self):
        with SolverFactory("scip_direct", solver_io="python") as opt:
            model = AbstractModel()
            model.X = Var(within=Integers)
            model.O = Objective(expr=model.X)

            instance = model.create_instance()
            results = opt.solve(instance)

            self.assertIn(
                results.solver.termination_condition,
                (
                    TerminationCondition.unbounded,
                    TerminationCondition.infeasibleOrUnbounded,
                ),
            )

    def test_optimal_mip(self):
        with SolverFactory("scip_direct", solver_io="python") as opt:
            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.O = Objective(expr=model.X)

            results = opt.solve(model, load_solutions=False)

            self.assertEqual(results.solution.status, SolutionStatus.optimal)


@unittest.skipIf(not scip_available, "The SCIP python bindings are not available")
class TestAddVar(unittest.TestCase):
    def test_add_single_variable(self):
        """Test that the variable is added correctly to `solver_model`."""
        model = ConcreteModel()

        opt = SolverFactory("scip_direct", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.getNVars(), 0)

        model.X = Var(within=Binary)

        opt._add_var(model.X)

        self.assertEqual(opt._solver_model.getNVars(), 1)
        self.assertEqual(opt._solver_model.getVars()[0].vtype(), "BINARY")

    def test_add_block_containing_single_variable(self):
        """Test that the variable is added correctly to `solver_model`."""
        model = ConcreteModel()

        opt = SolverFactory("scip_direct", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.getNVars(), 0)

        model.X = Var(within=Binary)

        opt._add_block(model)

        self.assertEqual(opt._solver_model.getNVars(), 1)
        self.assertEqual(opt._solver_model.getVars()[0].vtype(), "BINARY")

    def test_add_block_containing_multiple_variables(self):
        """Test that:
        - The variable is added correctly to `solver_model`
        - Fixed variable bounds are set correctly
        """
        model = ConcreteModel()

        opt = SolverFactory("scip_direct", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.getNVars(), 0)

        model.X1 = Var(within=Binary)
        model.X2 = Var(within=NonNegativeReals)
        model.X3 = Var(within=NonNegativeIntegers)

        model.X3.fix(5)

        opt._add_block(model)

        self.assertEqual(opt._solver_model.getNVars(), 3)
        scip_vars = opt._solver_model.getVars()
        vtypes = [scip_var.vtype() for scip_var in scip_vars]
        assert "BINARY" in vtypes and "CONTINUOUS" in vtypes and "INTEGER" in vtypes
        lbs = [scip_var.getLbGlobal() for scip_var in scip_vars]
        ubs = [scip_var.getUbGlobal() for scip_var in scip_vars]
        assert 0 in lbs and 5 in lbs
        assert (
            1 in ubs
            and 5 in ubs
            and any([opt._solver_model.isInfinity(ub) for ub in ubs])
        )


@unittest.skipIf(not scip_available, "The SCIP python bindings are not available")
class TestAddCon(unittest.TestCase):
    def test_add_single_constraint(self):
        model = ConcreteModel()
        model.X = Var(within=Binary)

        opt = SolverFactory("scip_direct", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.getNConss(), 0)

        model.C = Constraint(expr=model.X == 1)

        opt._add_constraint(model.C)

        self.assertEqual(opt._solver_model.getNConss(), 1)
        con = opt._solver_model.getConss()[0]
        self.assertEqual(con.isLinear(), 1)
        self.assertEqual(opt._solver_model.getRhs(con), 1)

    def test_add_block_containing_single_constraint(self):
        model = ConcreteModel()
        model.X = Var(within=Binary)

        opt = SolverFactory("scip_direct", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.getNConss(), 0)

        model.B = Block()
        model.B.C = Constraint(expr=model.X == 1)

        opt._add_block(model.B)

        self.assertEqual(opt._solver_model.getNConss(), 1)
        con = opt._solver_model.getConss()[0]
        self.assertEqual(con.isLinear(), 1)
        self.assertEqual(opt._solver_model.getRhs(con), 1)

    def test_add_block_containing_multiple_constraints(self):
        model = ConcreteModel()
        model.X = Var(within=Binary)

        opt = SolverFactory("scip_direct", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.getNConss(), 0)

        model.B = Block()
        model.B.C1 = Constraint(expr=model.X == 1)
        model.B.C2 = Constraint(expr=model.X <= 1)
        model.B.C3 = Constraint(expr=model.X >= 1)

        opt._add_block(model.B)

        self.assertEqual(opt._solver_model.getNConss(), 3)


@unittest.skipIf(not scip_available, "The SCIP python bindings are not available")
class TestLoadVars(unittest.TestCase):
    def setUp(self):
        opt = SolverFactory("scip_direct", solver_io="python")
        model = ConcreteModel()
        model.X = Var(within=NonNegativeReals, initialize=0)
        model.Y = Var(within=NonNegativeReals, initialize=0)

        model.C1 = Constraint(expr=2 * model.X + model.Y >= 8)
        model.C2 = Constraint(expr=model.X + 3 * model.Y >= 6)

        model.O = Objective(expr=model.X + model.Y)

        opt.solve(model, load_solutions=False, save_results=False)

        self._model = model
        self._opt = opt

    def test_all_vars_are_loaded(self):
        self.assertTrue(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertEqual(value(self._model.X), 0)
        self.assertEqual(value(self._model.Y), 0)

        self._opt.load_vars()

        self.assertFalse(self._model.X.stale)
        self.assertFalse(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertAlmostEqual(value(self._model.Y), 0.8)

    def test_only_specified_vars_are_loaded(self):
        self.assertTrue(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertEqual(value(self._model.X), 0)
        self.assertEqual(value(self._model.Y), 0)

        self._opt.load_vars([self._model.X])

        self.assertFalse(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertEqual(value(self._model.Y), 0)

        self._opt.load_vars([self._model.Y])

        self.assertFalse(self._model.X.stale)
        self.assertFalse(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertAlmostEqual(value(self._model.Y), 0.8)


if __name__ == "__main__":
    unittest.main()
