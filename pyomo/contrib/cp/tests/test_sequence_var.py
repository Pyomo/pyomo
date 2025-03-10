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
from pyomo.contrib.cp.interval_var import IntervalVar
from pyomo.contrib.cp.sequence_var import SequenceVar, IndexedSequenceVar
from pyomo.environ import ConcreteModel, Set


class TestScalarSequenceVar(unittest.TestCase):
    def test_initialize_with_no_data(self):
        m = ConcreteModel()
        m.i = SequenceVar()

        self.assertIsInstance(m.i, SequenceVar)
        self.assertIsInstance(m.i.interval_vars, list)
        self.assertEqual(len(m.i.interval_vars), 0)

        m.iv1 = IntervalVar()
        m.iv2 = IntervalVar()
        m.i.set_value(expr=[m.iv1, m.iv2])

        self.assertIsInstance(m.i.interval_vars, list)
        self.assertEqual(len(m.i.interval_vars), 2)
        self.assertIs(m.i.interval_vars[0], m.iv1)
        self.assertIs(m.i.interval_vars[1], m.iv2)

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
            """.strip(),
        )

    def test_interval_vars_not_a_list(self):
        m = self.get_model()

        with self.assertRaisesRegex(
            ValueError,
            "'expr' for SequenceVar must be a list of IntervalVars. "
            "Encountered type '<class 'int'>' constructing 'seq2'",
        ):
            m.seq2 = SequenceVar(expr=1)

    def test_interval_vars_list_includes_things_that_are_not_interval_vars(self):
        m = self.get_model()

        with self.assertRaisesRegex(
            ValueError,
            "The SequenceVar 'expr' argument must be a list of "
            "IntervalVars. The 'expr' for SequenceVar 'seq2' included "
            "an object of type '<class 'int'>'",
        ):
            m.seq2 = SequenceVar(expr=m.i)


class TestIndexedSequenceVar(unittest.TestCase):
    def test_initialize_with_not_data(self):
        m = ConcreteModel()
        m.i = SequenceVar([1, 2])

        self.assertIsInstance(m.i, IndexedSequenceVar)
        for j in [1, 2]:
            self.assertIsInstance(m.i[j].interval_vars, list)
            self.assertEqual(len(m.i[j].interval_vars), 0)

        m.iv = IntervalVar()
        m.iv2 = IntervalVar([0, 1])
        m.i[2] = [m.iv] + [m.iv2[i] for i in [0, 1]]

        self.assertEqual(len(m.i[2].interval_vars), 3)
        self.assertEqual(len(m.i[1].interval_vars), 0)
        self.assertIs(m.i[2].interval_vars[0], m.iv)
        for i in [0, 1]:
            self.assertIs(m.i[2].interval_vars[i + 1], m.iv2[i])

    def make_model(self):
        m = ConcreteModel()
        m.alphabetic = Set(initialize=['a', 'b'])
        m.numeric = Set(initialize=[1, 2])
        m.i = IntervalVar(m.alphabetic, m.numeric)

        def the_rule(m, j):
            return [m.i[j, k] for k in m.numeric]

        m.seq = SequenceVar(m.alphabetic, rule=the_rule)

        return m

    def test_initialize_with_rule(self):
        m = self.make_model()

        self.assertIsInstance(m.seq, IndexedSequenceVar)
        self.assertEqual(len(m.seq), 2)
        for j in m.alphabetic:
            self.assertTrue(j in m.seq)
            self.assertEqual(len(m.seq[j].interval_vars), 2)
            for k in m.numeric:
                self.assertIs(m.seq[j].interval_vars[k - 1], m.i[j, k])

    def test_pprint(self):
        m = self.make_model()
        m.seq.pprint()

        buf = StringIO()
        m.seq.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue().strip(),
            """
seq : Size=2, Index=alphabetic
    Key : IntervalVars
      a : [i[a,1], i[a,2]]
      b : [i[b,1], i[b,2]]""".strip(),
        )

    def test_multidimensional_index(self):
        m = self.make_model()

        @m.SequenceVar(m.alphabetic, m.numeric)
        def s(m, i, j):
            return [m.i[i, j]]

        self.assertIsInstance(m.s, IndexedSequenceVar)
        self.assertEqual(len(m.s), 4)
        for i in m.alphabetic:
            for j in m.numeric:
                self.assertTrue((i, j) in m.s)
                self.assertEqual(len(m.s[i, j].interval_vars), 1)
                self.assertIs(m.s[i, j].interval_vars[0], m.i[i, j])
