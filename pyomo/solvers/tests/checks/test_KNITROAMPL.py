# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import os

from pyomo.common import unittest
from pyomo.common.tempfiles import TempfileManager
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


class TestKNITROAMPLDefaultExecutable(unittest.TestCase):
    def test_default_executable_from_KNITRODIR(self):
        with TempfileManager.new_context() as tempfile:
            knitrodir = tempfile.create_tempdir()
            exe = os.path.join(knitrodir, 'knitroampl', 'knitroampl')
            # This makes a fake executable with the correct permissions
            # so KNITRO actually recognizes it
            os.mkdir(os.path.dirname(exe))
            with open(exe, 'w'):
                pass
            os.chmod(exe, 0o755)

            orig = os.environ.get('KNITRODIR')
            os.environ['KNITRODIR'] = knitrodir
            try:
                opt = SolverFactory('knitroampl')
                self.assertEqual(
                    os.path.realpath(opt._default_executable()), os.path.realpath(exe)
                )
            finally:
                if orig is None:
                    del os.environ['KNITRODIR']
                else:
                    os.environ['KNITRODIR'] = orig


@unittest.skipIf(not knitroampl_available, "The 'knitroampl' command is not available")
@unittest.pytest.mark.solver("knitroampl")
class TestKNITROAMPLInterface(unittest.TestCase):
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
