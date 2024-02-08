#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
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
        ref = """\\* Source Pyomo model name=unknown *\\

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
