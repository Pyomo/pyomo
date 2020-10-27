#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

import pyomo.environ
from pyomo.core import (ConcreteModel, Var, Objective,
                        Constraint, NonNegativeReals)
from pyomo.opt import SolverFactory

try:
    import cplex

    cplexpy_available = True
except ImportError:
    cplexpy_available = False


@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestQuadraticObjective(unittest.TestCase):
    def test_quadratic_objective_is_set(self):
        model = ConcreteModel()
        model.X = Var(bounds=(-2, 2))
        model.Y = Var(bounds=(-2, 2))
        model.O = Objective(expr=model.X ** 2 + model.Y ** 2)
        model.C1 = Constraint(expr=model.Y >= 2 * model.X - 1)
        model.C2 = Constraint(expr=model.Y >= -model.X + 2)
        opt = SolverFactory("cplex_persistent")
        opt.set_instance(model)
        opt.solve()

        self.assertAlmostEqual(model.X.value, 1, places=3)
        self.assertAlmostEqual(model.Y.value, 1, places=3)

        del model.O
        model.O = Objective(expr=model.X ** 2)
        opt.set_objective(model.O)
        opt.solve()
        self.assertAlmostEqual(model.X.value, 0, places=3)
        self.assertAlmostEqual(model.Y.value, 2, places=3)

    def test_add_column(self):
        m = ConcreteModel()
        m.x = Var(within=NonNegativeReals)
        m.c = Constraint(expr=(0, m.x, 1))
        m.obj = Objective(expr=-m.x)

        opt = SolverFactory('cplex_persistent')
        opt.set_instance(m)
        opt.solve()
        self.assertAlmostEqual(m.x.value, 1)

        m.y = Var(within=NonNegativeReals)

        opt.add_column(m, m.y, -3, [m.c], [2])
        opt.solve()

        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 0.5)

    def test_add_column_exceptions(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=(0, m.x, 1))
        m.ci = Constraint([1,2], rule=lambda m,i:(0,m.x,i+1))
        m.cd = Constraint(expr=(0, -m.x, 1))
        m.cd.deactivate()
        m.obj = Objective(expr=-m.x)

        opt = SolverFactory('cplex_persistent')

        # set_instance not called
        self.assertRaises(RuntimeError, opt.add_column, m, m.x, 0, [m.c], [1])

        opt.set_instance(m)

        m2 = ConcreteModel()
        m2.y = Var()
        m2.c = Constraint(expr=(0,m.x,1))

        # different model than attached to opt
        self.assertRaises(RuntimeError, opt.add_column, m2, m2.y, 0, [], [])
        # pyomo var attached to different model
        self.assertRaises(RuntimeError, opt.add_column, m, m2.y, 0, [], [])

        z = Var()
        # pyomo var floating
        self.assertRaises(RuntimeError, opt.add_column, m, z, -2, [m.c, z], [1])

        m.y = Var()
        # len(coefficents) == len(constraints)
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1,2])
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c, z], [1])

        # add indexed constraint
        self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.ci], [1])
        # add something not a _ConstraintData
        self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.x], [1])

        # constraint not on solver model
        self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m2.c], [1])

        # inactive constraint
        self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m.cd], [1])

        opt.add_var(m.y)
        # var already in solver model
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1])
