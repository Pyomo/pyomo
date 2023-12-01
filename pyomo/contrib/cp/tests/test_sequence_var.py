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
from pyomo.contrib.cp.interval_var import IntervalVar
from pyomo.contrib.cp.sequence_var import SequenceVar, IndexedSequenceVar
from pyomo.environ import ConcreteModel, Integers, Set, value, Var


class TestScalarSequenceVar(unittest.TestCase):
    def test_initialize_with_no_data(self):
        m = ConcreteModel()
        m.i = SequenceVar()

        self.assertIsInstance(m.i, SequenceVar)
        self.assertIsInstance(m.i.interval_vars, list)
        self.assertEqual(len(m.i.interval_vars), 0)

    def get_model(self):
        m = ConcreteModel()
        m.S = Set(initialize=range(3))
        m.i = IntervalVar(m.S, start=(0, 5))
        m.seq = SequenceVar(expr=[m.i[j] for j in m.S])

        return m

    def test_initialize_with_expr(self):
        m = self.get_model()
        self.assertEqual(len(m.seq.interval_vars), 3)
        for j in m.S:
            self.assertIs(m.seq.interval_vars[j], m.i[j])

    def test_pprint(self):
        m = self.get_model()
        buf = StringIO()
        m.seq.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue().strip(),
            """
seq : Size=1, Index=None
    Key  : IntervalVars
    None : [i[0], i[1], i[2]]
            """.strip()
        )

class TestIndexedSequenceVar(unittest.TestCase):
    def test_initialize_with_not_data(self):
        m = ConcreteModel()
        m.i = SequenceVar([1, 2])

        self.assertIsInstance(m.i, IndexedSequenceVar)
        for j in [1, 2]:
            self.assertIsInstance(m.i[j].interval_vars, list)
            self.assertEqual(len(m.i[j].interval_vars), 0)

    def make_model(self):
        m = ConcreteModel()
        m.alph = Set(initialize=['a', 'b'])
        m.num = Set(initialize=[1, 2])
        m.i = IntervalVar(m.alph, m.num)

        def the_rule(m, j):
            return [m.i[j, k] for k in m.num]
        m.seq = SequenceVar(m.alph, rule=the_rule)

        return m

    def test_initialize_with_rule(self):
        m = self.make_model()

        self.assertIsInstance(m.seq, IndexedSequenceVar)
        self.assertEqual(len(m.seq), 2)
        for j in m.alph:
            self.assertTrue(j in m.seq)
            self.assertEqual(len(m.seq[j].interval_vars), 2)
            for k in m.num:
                self.assertIs(m.seq[j].interval_vars[k - 1], m.i[j, k])

    def test_pprint(self):
        m = self.make_model()
        m.seq.pprint()

        buf = StringIO()
        m.seq.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue().strip(),
            """
seq : Size=2, Index=alph
    Key : IntervalVars
      a : [i[a,1], i[a,2]]
      b : [i[b,1], i[b,2]]""".strip()
        )
