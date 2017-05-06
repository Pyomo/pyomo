#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for Discrete Variable Transformations

import pyutilib.th as unittest

from pyomo.environ import *

solvers = pyomo.opt.check_available_solvers('cplex', 'gurobi', 'glpk')

def _generateModel():
    model = ConcreteModel()
    model.x = Var(within=Binary)
    model.y = Var()
    model.c1 = Constraint(expr=model.y >= model.x)
    model.c2 = Constraint(expr=model.y >= 1.5-model.x)
    model.obj = Objective(expr=model.y)
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    return model

class Test(unittest.TestCase):

    @unittest.skipIf( len(solvers) == 0, "LP/MIP solver not available")
    def test_solve_relax_transform(self):
        s = SolverFactory(solvers[0])
        m = _generateModel()
        self.assertIs(m.x.domain, Binary)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        s.solve(m)
        self.assertEqual(len(m.dual), 0)

        TransformationFactory('core.relax_discrete').apply_to(m)
        self.assertIs(m.x.domain, NonNegativeReals)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        s.solve(m)
        self.assertEqual(len(m.dual), 2)
        self.assertAlmostEqual(m.dual[m.c1], -0.5, 4)
        self.assertAlmostEqual(m.dual[m.c2], -0.5, 4)


    @unittest.skipIf( len(solvers) == 0, "LP/MIP solver not available")
    def test_solve_fix_transform(self):
        s = SolverFactory(solvers[0])
        m = _generateModel()
        self.assertIs(m.x.domain, Binary)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        s.solve(m)
        m.pprint()
        self.assertEqual(len(m.dual), 0)

        TransformationFactory('core.fix_discrete').apply_to(m)
        self.assertIs(m.x.domain, Binary)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        s.solve(m)
        self.assertEqual(len(m.dual), 2)
        self.assertAlmostEqual(m.dual[m.c1], -1, 4)
        self.assertAlmostEqual(m.dual[m.c2], 0, 4)

if __name__ == "__main__":
    unittest.main()
