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
    """
    Borrowed from examples/pyomo/sos/sos2_piecewise.py
    """
    model = pyo.ConcreteModel()

    model.idx_set = pyo.Set(initialize=[1, 2])
    DOMAIN_PTS = {1: [1, 2, 3], 2: [1, 2, 3]}
    F = {1: [1, 4, 9], 2: [1, 4, 9]}
    # Note we can also implement this like below
    # F = lambda x: x**2
    # Update the return value for constraint2_rule if
    # F is defined using the function above

    # Indexing set required for the SOSConstraint declaration
    def SOS_indices_init(model, t):
        return [(t, i) for i in range(len(DOMAIN_PTS[t]))]

    model.SOS_indices = pyo.Set(
        model.idx_set, dimen=2, ordered=True, initialize=SOS_indices_init
    )

    def sos_var_indices_init(model):
        return [(t, i) for t in model.idx_set for i in range(len(DOMAIN_PTS[t]))]

    model.sos_var_indices = pyo.Set(
        ordered=True, dimen=2, initialize=sos_var_indices_init
    )

    model.x = pyo.Var(model.idx_set)  # domain variable
    model.Fx = pyo.Var(model.idx_set)  # range variable
    model.y = pyo.Var(
        model.sos_var_indices, within=pyo.NonNegativeReals
    )  # SOS2 variable

    model.obj = pyo.Objective(expr=pyo.sum_product(model.Fx), sense=pyo.maximize)

    def constraint1_rule(model, t):
        return model.x[t] == sum(
            model.y[t, i] * DOMAIN_PTS[t][i] for i in range(len(DOMAIN_PTS[t]))
        )

    def constraint2_rule(model, t):
        # Uncomment below for F defined as dictionary
        return model.Fx[t] == sum(
            model.y[t, i] * F[t][i] for i in range(len(DOMAIN_PTS[t]))
        )
        # Uncomment below for F defined as lambda function
        # return model.Fx[t] == sum(model.y[t,i]*F(DOMAIN_PTS[t][i]) for i in range(len(DOMAIN_PTS[t])) )

    def constraint3_rule(model, t):
        return sum(model.y[t, j] for j in range(len(DOMAIN_PTS[t]))) == 1

    model.constraint1 = pyo.Constraint(model.idx_set, rule=constraint1_rule)
    model.constraint2 = pyo.Constraint(model.idx_set, rule=constraint2_rule)
    model.constraint3 = pyo.Constraint(model.idx_set, rule=constraint3_rule)
    model.SOS_set_constraint = pyo.SOSConstraint(
        model.idx_set, var=model.y, index=model.SOS_indices, sos=2
    )

    # Fix the answer for testing purposes
    model.set_answer_constraint1 = pyo.Constraint(expr=model.x[1] == 2.5)
    model.set_answer_constraint2 = pyo.Constraint(expr=model.x[2] == 2.0)
    return model


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

    def test_SOS_ordering_bug(self):
        # This is in response to a bug that was identified with
        # config.row_order with SOS constraints. If an SOS constraint
        # was added and the `config.row_order` option selected,
        # bug would ensue.
        ref_no_row_order = r"""\* Source Pyomo model name=unknown *\

max 
x1:
+1 x2
+1 x3

s.t.

c_e_x4_:
+1 x5
-1 x6
-2 x7
-3 x8
= 0

c_e_x9_:
+1 x10
-1 x11
-2 x12
-3 x13
= 0

c_e_x14_:
+1 x2
-1 x6
-4 x7
-9 x8
= 0

c_e_x15_:
+1 x3
-1 x11
-4 x12
-9 x13
= 0

c_e_x16_:
+1 x6
+1 x7
+1 x8
= 1

c_e_x17_:
+1 x11
+1 x12
+1 x13
= 1

c_e_x18_:
+1 x5
= 2.5

c_e_x19_:
+1 x10
= 2.0

bounds
   -inf <= x2 <= +inf
   -inf <= x3 <= +inf
   -inf <= x5 <= +inf
   -inf <= x10 <= +inf
   0 <= x6 <= +inf
   0 <= x7 <= +inf
   0 <= x8 <= +inf
   0 <= x11 <= +inf
   0 <= x12 <= +inf
   0 <= x13 <= +inf
SOS

x20: S2::
  x6:1
  x7:2
  x8:3

x21: S2::
  x11:1
  x12:2
  x13:3

end
"""
        ref_row_order = r"""\* Source Pyomo model name=unknown *\

max 
x1:
+1 x2
+1 x3

s.t.

c_e_x4_:
+1 x5
+1 x6
+1 x7
= 1

c_e_x8_:
+1 x9
+1 x10
+1 x11
= 1

c_e_x12_:
-1 x5
-2 x6
-3 x7
+1 x13
= 0

c_e_x14_:
-1 x9
-2 x10
-3 x11
+1 x15
= 0

c_e_x16_:
+1 x2
-1 x5
-4 x6
-9 x7
= 0

c_e_x17_:
+1 x3
-1 x9
-4 x10
-9 x11
= 0

c_e_x18_:
+1 x13
= 2.5

c_e_x19_:
+1 x15
= 2.0

bounds
   -inf <= x2 <= +inf
   -inf <= x3 <= +inf
   0 <= x5 <= +inf
   0 <= x6 <= +inf
   0 <= x7 <= +inf
   0 <= x9 <= +inf
   0 <= x10 <= +inf
   0 <= x11 <= +inf
   -inf <= x13 <= +inf
   -inf <= x15 <= +inf
SOS

x20: S2::
  x5:1
  x6:2
  x7:3

x21: S2::
  x9:1
  x10:2
  x11:3

end
"""
        model = create_sos_model()
        with LoggingIntercept() as OUT1:
            LPWriter().write(model, OUT1)
        self.assertEqual(OUT1.getvalue(), ref_no_row_order)
        with LoggingIntercept() as OUT2:
            LPWriter().write(model, OUT2, row_order=[model.constraint3])
        self.assertEqual(OUT2.getvalue(), ref_row_order)
