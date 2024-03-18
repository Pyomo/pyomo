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

import pyomo.environ
import pyomo.common.unittest as unittest

from pyomo.core import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeReals,
    NonNegativeIntegers,
    Reals,
    Binary,
    SOSConstraint,
    Set,
    sin,
    cos,
    exp,
    log,
)
from pyomo.opt import SolverFactory

try:
    import pyscipopt

    scip_available = True
except ImportError:
    scip_available = False


@unittest.skipIf(not scip_available, "The SCIP python bindings are not available")
class TestQuadraticObjective(unittest.TestCase):
    def test_quadratic_objective_linear_surrogate_is_set(self):
        m = ConcreteModel()
        m.X = Var(bounds=(-2, 2))
        m.Y = Var(bounds=(-2, 2))
        m.Z = Var(within=Reals)
        m.O = Objective(expr=m.Z)
        m.C1 = Constraint(expr=m.Y >= 2 * m.X - 1)
        m.C2 = Constraint(expr=m.Y >= -m.X + 2)
        m.C3 = Constraint(expr=m.Z >= m.X**2 + m.Y**2)
        opt = SolverFactory("scip_persistent")
        opt.set_instance(m)
        opt.solve()

        self.assertAlmostEqual(m.X.value, 1, places=3)
        self.assertAlmostEqual(m.Y.value, 1, places=3)

        opt.reset()

        opt.remove_constraint(m.C3)
        del m.C3
        m.C3 = Constraint(expr=m.Z >= m.X**2)
        opt.add_constraint(m.C3)
        opt.solve()
        self.assertAlmostEqual(m.X.value, 0, places=3)
        self.assertAlmostEqual(m.Y.value, 2, places=3)

    def test_add_and_remove_sos(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.X = Var(m.I, bounds=(-2, 2))

        m.C = SOSConstraint(var=m.X, sos=1)

        m.O = Objective(expr=m.X[1] + m.X[2])

        opt = SolverFactory("scip_persistent")

        opt.set_instance(m)
        opt.solve()

        zero_val_var = 0
        for i in range(1, 4):
            if -0.001 < m.X[i].value < 0.001:
                zero_val_var += 1
        assert zero_val_var == 2

        opt.reset()

        opt.remove_sos_constraint(m.C)
        del m.C

        m.C = SOSConstraint(var=m.X, sos=2)
        opt.add_sos_constraint(m.C)

        opt.solve()

        zero_val_var = 0
        for i in range(1, 4):
            if -0.001 < m.X[i].value < 0.001:
                zero_val_var += 1
        assert zero_val_var == 1

    def test_get_and_set_param(self):
        m = ConcreteModel()
        m.X = Var(bounds=(-2, 2))
        m.O = Objective(expr=m.X)
        m.C3 = Constraint(expr=m.X <= 2)
        opt = SolverFactory("scip_persistent")
        opt.set_instance(m)

        opt.set_scip_param("limits/time", 60)

        assert opt.get_scip_param("limits/time") == 60

    def test_non_linear(self):

        PI = 3.141592653589793238462643
        NWIRES = 11
        DIAMETERS = [
            0.207,
            0.225,
            0.244,
            0.263,
            0.283,
            0.307,
            0.331,
            0.362,
            0.394,
            0.4375,
            0.500,
        ]
        PRELOAD = 300.0
        MAXWORKLOAD = 1000.0
        MAXDEFLECT = 6.0
        DEFLECTPRELOAD = 1.25
        MAXFREELEN = 14.0
        MAXCOILDIAM = 3.0
        MAXSHEARSTRESS = 189000.0
        SHEARMOD = 11500000.0

        m = ConcreteModel()
        m.coil = Var(within=NonNegativeReals)
        m.wire = Var(within=NonNegativeReals)
        m.defl = Var(
            bounds=(DEFLECTPRELOAD / (MAXWORKLOAD - PRELOAD), MAXDEFLECT / PRELOAD)
        )
        m.ncoils = Var(within=NonNegativeIntegers)
        m.const1 = Var(within=NonNegativeReals)
        m.const2 = Var(within=NonNegativeReals)
        m.volume = Var(within=NonNegativeReals)
        m.I = Set(initialize=[i for i in range(NWIRES)])
        m.y = Var(m.I, within=Binary)

        m.O = Objective(expr=m.volume)

        m.c1 = Constraint(
            expr=PI / 2 * (m.ncoils + 2) * m.coil * m.wire**2 - m.volume == 0
        )

        m.c2 = Constraint(expr=m.coil / m.wire - m.const1 == 0)

        m.c3 = Constraint(
            expr=(4 * m.const1 - 1) / (4 * m.const1 - 4) + 0.615 / m.const1 - m.const2
            == 0
        )

        m.c4 = Constraint(
            expr=8.0 * MAXWORKLOAD / PI * m.const1 * m.const2
            - MAXSHEARSTRESS * m.wire**2
            <= 0
        )

        m.c5 = Constraint(
            expr=8 / SHEARMOD * m.ncoils * m.const1**3 / m.wire - m.defl == 0
        )

        m.c6 = Constraint(
            expr=MAXWORKLOAD * m.defl + 1.05 * m.ncoils * m.wire + 2.1 * m.wire
            <= MAXFREELEN
        )

        m.c7 = Constraint(expr=m.coil + m.wire <= MAXCOILDIAM)

        m.c8 = Constraint(
            expr=sum(m.y[i] * DIAMETERS[i] for i in range(NWIRES)) - m.wire == 0
        )

        m.c9 = Constraint(expr=sum(m.y[i] for i in range(NWIRES)) == 1)

        opt = SolverFactory("scip_persistent")
        opt.set_instance(m)

        opt.solve()

        self.assertAlmostEqual(m.volume.value, 1.6924910128, places=2)

    def test_non_linear_unary_expressions(self):

        m = ConcreteModel()
        m.X = Var(bounds=(1, 2))
        m.Y = Var(within=Reals)

        m.O = Objective(expr=m.Y)

        m.C = Constraint(expr=exp(m.X) == m.Y)

        opt = SolverFactory("scip_persistent")
        opt.set_instance(m)

        opt.solve()
        self.assertAlmostEqual(m.X.value, 1, places=3)
        self.assertAlmostEqual(m.Y.value, exp(1), places=3)

        opt.reset()
        opt.remove_constraint(m.C)
        del m.C

        m.C = Constraint(expr=log(m.X) == m.Y)
        opt.add_constraint(m.C)
        opt.solve()
        self.assertAlmostEqual(m.X.value, 1, places=3)
        self.assertAlmostEqual(m.Y.value, 0, places=3)

        opt.reset()
        opt.remove_constraint(m.C)
        del m.C

        m.C = Constraint(expr=sin(m.X) == m.Y)
        opt.add_constraint(m.C)
        opt.solve()
        self.assertAlmostEqual(m.X.value, 1, places=3)
        self.assertAlmostEqual(m.Y.value, sin(1), places=3)

        opt.reset()
        opt.remove_constraint(m.C)
        del m.C

        m.C = Constraint(expr=cos(m.X) == m.Y)
        opt.add_constraint(m.C)
        opt.solve()
        self.assertAlmostEqual(m.X.value, 2, places=3)
        self.assertAlmostEqual(m.Y.value, cos(2), places=3)

    def test_add_column(self):
        m = ConcreteModel()
        m.x = Var(within=NonNegativeReals)
        m.c = Constraint(expr=(0, m.x, 1))
        m.obj = Objective(expr=-m.x)

        opt = SolverFactory("scip_persistent")
        opt.set_instance(m)
        opt.solve()
        self.assertAlmostEqual(m.x.value, 1)

        m.y = Var(within=NonNegativeReals)

        opt.reset()

        opt.add_column(m, m.y, -3, [m.c], [2])
        opt.solve()

        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 0.5)

    def test_add_column_exceptions(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=(0, m.x, 1))
        m.ci = Constraint([1, 2], rule=lambda m, i: (0, m.x, i + 1))
        m.cd = Constraint(expr=(0, -m.x, 1))
        m.cd.deactivate()
        m.obj = Objective(expr=-m.x)

        opt = SolverFactory("scip_persistent")

        # set_instance not called
        self.assertRaises(RuntimeError, opt.add_column, m, m.x, 0, [m.c], [1])

        opt.set_instance(m)

        m2 = ConcreteModel()
        m2.y = Var()
        m2.c = Constraint(expr=(0, m.x, 1))

        # different model than attached to opt
        self.assertRaises(RuntimeError, opt.add_column, m2, m2.y, 0, [], [])
        # pyomo var attached to different model
        self.assertRaises(RuntimeError, opt.add_column, m, m2.y, 0, [], [])

        z = Var()
        # pyomo var floating
        self.assertRaises(RuntimeError, opt.add_column, m, z, -2, [m.c, z], [1])

        m.y = Var()
        # len(coefficients) == len(constraints)
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1, 2])
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


if __name__ == "__main__":
    unittest.main()
