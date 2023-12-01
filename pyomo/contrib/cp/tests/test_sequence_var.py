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

    def test_initialize_with_expr(self):
        m = ConcreteModel()
        m.S = Set(initialize=range(3))
        m.i = IntervalVar(m.S, start=(0, 5))
        m.seq = SequenceVar(expr=[m.i[j] for j in m.S])
        self.assertEqual(len(m.seq.interval_vars), 3)
        for j in m.S:
            self.assertIs(m.seq.interval_vars[j], m.i[j])


class TestIndexedSequenceVar(unittest.TestCase):
    def test_initialize_with_rule(self):
        m = ConcreteModel()
        m.alph = Set(initialize=['a', 'b'])
        m.num = Set(initialize=[1, 2])
        m.i = IntervalVar(m.alph, m.num)

        def the_rule(m, j):
            return [m.i[j, k] for k in m.num]
        m.seq = SequenceVar(m.alph, rule=the_rule)
        m.seq.pprint()

        self.assertIsInstance(m.seq, IndexedSequenceVar)
        self.assertEqual(len(m.seq), 2)
        for j in m.alph:
            self.assertTrue(j in m.seq)
            self.assertEqual(len(m.seq[j].interval_vars), 2)
            for k in m.num:
                self.assertIs(m.seq[j].interval_vars[k - 1], m.i[j, k])

