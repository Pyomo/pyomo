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

from pyomo.common import unittest
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    Suffix,
    NonNegativeIntegers,
    NonNegativeReals,
    value,
)
from pyomo.opt import SolverFactory, TerminationCondition

knitroampl_available = SolverFactory('knitroampl').available(False)


class TestKNITROAMPLInterface(unittest.TestCase):
    @unittest.skipIf(
        not knitroampl_available, "The 'knitroampl' command is not available"
    )
    def test_infeasible_lp(self):
        with SolverFactory('knitroampl') as opt:
            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.C1 = Constraint(expr=model.X == 1)
            model.C2 = Constraint(expr=model.X == 2)
            model.Obj = Objective(expr=model.X)

            results = opt.solve(model)

            self.assertEqual(
                results.solver.termination_condition, TerminationCondition.infeasible
            )

    @unittest.skipIf(
        not knitroampl_available, "The 'knitroampl' command is not available"
    )
    def test_unbounded_lp(self):
        with SolverFactory('knitroampl') as opt:
            model = ConcreteModel()
            model.X = Var()
            model.Obj = Objective(expr=model.X)

            results = opt.solve(model)

            self.assertIn(
                results.solver.termination_condition,
                (
                    TerminationCondition.unbounded,
                    TerminationCondition.infeasibleOrUnbounded,
                ),
            )

    @unittest.skipIf(
        not knitroampl_available, "The 'knitroampl' command is not available"
    )
    def test_optimal_lp(self):
        with SolverFactory('knitroampl') as opt:
            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.C1 = Constraint(expr=model.X >= 2.5)
            model.Obj = Objective(expr=model.X)

            results = opt.solve(model, load_solutions=True)

            self.assertEqual(
                results.solver.termination_condition, TerminationCondition.optimal
            )
            self.assertAlmostEqual(value(model.X), 2.5)

    @unittest.skipIf(
        not knitroampl_available, "The 'knitroampl' command is not available"
    )
    def test_get_duals_lp(self):
        with SolverFactory('knitroampl') as opt:
            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.Y = Var(within=NonNegativeReals)

            model.C1 = Constraint(expr=2 * model.X + model.Y >= 8)
            model.C2 = Constraint(expr=model.X + 3 * model.Y >= 6)

            model.Obj = Objective(expr=model.X + model.Y)

            results = opt.solve(model, suffixes=['dual'], load_solutions=False)

            model.dual = Suffix(direction=Suffix.IMPORT)
            model.solutions.load_from(results)

            self.assertAlmostEqual(model.dual[model.C1], 0.4)
            self.assertAlmostEqual(model.dual[model.C2], 0.2)

    @unittest.skipIf(
        not knitroampl_available, "The 'knitroampl' command is not available"
    )
    def test_infeasible_mip(self):
        with SolverFactory('knitroampl') as opt:
            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.C1 = Constraint(expr=model.X == 1)
            model.C2 = Constraint(expr=model.X == 2)
            model.Obj = Objective(expr=model.X)

            results = opt.solve(model)

            self.assertEqual(
                results.solver.termination_condition, TerminationCondition.infeasible
            )

    @unittest.skipIf(
        not knitroampl_available, "The 'knitroampl' command is not available"
    )
    def test_optimal_mip(self):
        with SolverFactory('knitroampl') as opt:
            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.C1 = Constraint(expr=model.X >= 2.5)
            model.Obj = Objective(expr=model.X)

            results = opt.solve(model, load_solutions=True)

            self.assertEqual(
                results.solver.termination_condition, TerminationCondition.optimal
            )
            self.assertAlmostEqual(value(model.X), 3)


if __name__ == "__main__":
    unittest.main()
