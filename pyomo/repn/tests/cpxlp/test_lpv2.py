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

from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.log import LoggingIntercept

import pyomo.environ as pyo

from pyomo.repn.plugins.lp_writer import LPWriter


def create_sos_model():
    m = pyo.ConcreteModel()

    m.IDX = pyo.Set(initialize=[1, 2])
    m.SOS2_VARS = pyo.Set(m.IDX, initialize={1: [1, 2, 3], 2: [4, 5, 6]})
    m.SOS3_VARS = pyo.Set(m.IDX, initialize={1: [1, 3, 5], 2: [2, 4, 6]})

    m.x = pyo.Var(pyo.RangeSet(6))

    m.sos1 = pyo.SOSConstraint(var=m.x, sos=2)
    m.sos2 = pyo.SOSConstraint(m.IDX, var=m.x, index=m.SOS2_VARS, sos=1)
    m.sos3 = pyo.SOSConstraint(m.IDX, var=m.x, index=m.SOS3_VARS, sos=2)

    @m.Constraint()
    def con1(m, i):
        return sum(m.x[i] for i in m.x) == 0

    @m.Constraint(m.IDX)
    def con2(m, i):
        return m.x[i] >= 20 + i

    @m.Constraint(m.IDX)
    def con3(m, i):
        return m.x[i] >= 30 + i

    return m


class TestLPv2(unittest.TestCase):
    def test_warn_export_suffixes(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x)
        m.con = pyo.Constraint(expr=m.x >= 2)
        m.b = pyo.Block()
        m.ignored = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.duals = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        m.b.duals = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        m.b.scaling = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        # Empty suffixes are ignored
        writer = LPWriter()
        with LoggingIntercept() as LOG:
            writer.write(m, StringIO())
        self.assertEqual(LOG.getvalue(), "")

        # Import are ignored, export and import/export are warned
        m.duals[m.con] = 5
        m.ignored[m.x] = 6
        m.b.scaling[m.x] = 7

        writer = LPWriter()
        with LoggingIntercept() as LOG:
            writer.write(m, StringIO())
        self.assertEqual(
            LOG.getvalue(),
            """EXPORT Suffix 'duals' found on 1 block:
    duals
LP writer cannot export suffixes to LP files.  Skipping.
EXPORT Suffix 'scaling' found on 1 block:
    b.scaling
LP writer cannot export suffixes to LP files.  Skipping.
""",
        )

        # Counting works correctly
        m.b.duals[m.x] = 7

        writer = LPWriter()
        with LoggingIntercept() as LOG:
            writer.write(m, StringIO())
        self.assertEqual(
            LOG.getvalue(),
            """EXPORT Suffix 'duals' found on 2 blocks:
    duals
    b.duals
LP writer cannot export suffixes to LP files.  Skipping.
EXPORT Suffix 'scaling' found on 1 block:
    b.scaling
LP writer cannot export suffixes to LP files.  Skipping.
""",
        )

    def test_deterministic_unordered_sets(self):
        ref = r"""\* Source Pyomo model name=unknown *\

min 
o:
+1 x(a)
+1 x(aaaaa)
+1 x(ooo)
+1 x(z)

s.t.

c_l_c(a)_:
+1 x(a)
>= 1

c_l_c(aaaaa)_:
+1 x(aaaaa)
>= 5

c_l_c(ooo)_:
+1 x(ooo)
>= 3

c_l_c(z)_:
+1 x(z)
>= 1

bounds
   -inf <= x(a) <= +inf
   -inf <= x(aaaaa) <= +inf
   -inf <= x(ooo) <= +inf
   -inf <= x(z) <= +inf
end
"""
        set_init = ['a', 'z', 'ooo', 'aaaaa']

        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=set_init, ordered=False)
        m.x = pyo.Var(m.I)
        m.c = pyo.Constraint(m.I, rule=lambda m, i: m.x[i] >= len(i))
        m.o = pyo.Objective(expr=sum(m.x[i] for i in m.I))

        OUT = StringIO()
        with LoggingIntercept() as LOG:
            LPWriter().write(m, OUT, symbolic_solver_labels=True)
        self.assertEqual(LOG.getvalue(), "")
        print(OUT.getvalue())
        self.assertEqual(ref, OUT.getvalue())

        m = pyo.ConcreteModel()
        m.I = pyo.Set()
        m.x = pyo.Var(pyo.Any)
        m.c = pyo.Constraint(pyo.Any)
        for i in set_init:
            m.c[i] = m.x[i] >= len(i)
        m.o = pyo.Objective(expr=sum(m.x.values()))

        OUT = StringIO()
        with LoggingIntercept() as LOG:
            LPWriter().write(m, OUT, symbolic_solver_labels=True)
        self.assertEqual(LOG.getvalue(), "")

        self.assertEqual(ref, OUT.getvalue())

    def test_SOS_ordering(self):
        # This is in response to a bug that was identified with
        # config.row_order with SOS constraints. If an SOS constraint
        # was added and the `config.row_order` option selected,
        # bug would ensue.

        model = create_sos_model()

        OUT = StringIO()
        LPWriter().write(model, OUT, symbolic_solver_labels=True)
        self.assertEqual(
            r"""\* Source Pyomo model name=unknown *\

min 
ScalarObjective:
+1.0 ONE_VAR_CONSTANT

s.t.

c_e_con1_:
+1 x(1)
+1 x(2)
+1 x(3)
+1 x(4)
+1 x(5)
+1 x(6)
= 0

c_l_con2(1)_:
+1 x(1)
>= 21

c_l_con2(2)_:
+1 x(2)
>= 22

c_l_con3(1)_:
+1 x(1)
>= 31

c_l_con3(2)_:
+1 x(2)
>= 32

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   -inf <= x(1) <= +inf
   -inf <= x(2) <= +inf
   -inf <= x(3) <= +inf
   -inf <= x(4) <= +inf
   -inf <= x(5) <= +inf
   -inf <= x(6) <= +inf
SOS

sos1: S2::
  x(1):1
  x(2):2
  x(3):3
  x(4):4
  x(5):5
  x(6):6

sos2(1): S1::
  x(1):1
  x(2):2
  x(3):3

sos2(2): S1::
  x(4):1
  x(5):2
  x(6):3

sos3(1): S2::
  x(1):1
  x(3):2
  x(5):3

sos3(2): S2::
  x(2):1
  x(4):2
  x(6):3

end
""",
            OUT.getvalue(),
        )

        OUT = StringIO()
        LPWriter().write(
            model,
            OUT,
            symbolic_solver_labels=True,
            row_order=[
                model.sos3[2],
                model.con3[2],
                model.con2,
                model.sos2,
                model.sos1,
                model.con1,
            ],
        )
        self.assertEqual(
            r"""\* Source Pyomo model name=unknown *\

min 
ScalarObjective:
+1.0 ONE_VAR_CONSTANT

s.t.

c_l_con3(2)_:
+1 x(2)
>= 32

c_l_con2(1)_:
+1 x(1)
>= 21

c_l_con2(2)_:
+1 x(2)
>= 22

c_e_con1_:
+1 x(1)
+1 x(2)
+1 x(3)
+1 x(4)
+1 x(5)
+1 x(6)
= 0

c_l_con3(1)_:
+1 x(1)
>= 31

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   -inf <= x(1) <= +inf
   -inf <= x(2) <= +inf
   -inf <= x(3) <= +inf
   -inf <= x(4) <= +inf
   -inf <= x(5) <= +inf
   -inf <= x(6) <= +inf
SOS

sos3(2): S2::
  x(2):1
  x(4):2
  x(6):3

sos2(1): S1::
  x(1):1
  x(2):2
  x(3):3

sos2(2): S1::
  x(4):1
  x(5):2
  x(6):3

sos1: S2::
  x(1):1
  x(2):2
  x(3):3
  x(4):4
  x(5):5
  x(6):6

sos3(1): S2::
  x(1):1
  x(3):2
  x(5):3

end
""",
            OUT.getvalue(),
        )
