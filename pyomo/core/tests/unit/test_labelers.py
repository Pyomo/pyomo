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
from pyomo.environ import ConcreteModel, Var, RangeSet, Block, Constraint, CounterLabeler, NumericLabeler, TextLabeler, ComponentUID, ShortNameLabeler, CNameLabeler, CuidLabeler, AlphaNumericTextLabeler, NameLabeler


class LabelerTests(unittest.TestCase):

    def setUp(self):
        m = ConcreteModel()
        m.mycomp = Var()
        m.MyComp = Var()
        m.that = Var()
        self.long1 = m.myverylongcomponentname = Var()
        self.long2 = m.myverylongcomponentnamerighthere = Var()
        self.long3 = m.anotherlongonebutdifferent = Var()
        self.long4 = m.anotherlongonebutdifferentlongcomponentname = Var()
        self.long5 = m.longcomponentname_1_ = Var()
        m.s = RangeSet(10)
        m.ind = Var(m.s)
        m.myblock = Block()
        m.myblock.mystreet = Constraint()
        m.add_component("myblock.mystreet", Var())
        self.thecopy = m.__getattribute__("myblock.mystreet")
        self.m = m

    def test_counterlabeler(self):
        m = self.m
        lbl = CounterLabeler()
        self.assertEqual(lbl(m.mycomp), 1)
        self.assertEqual(lbl(m.mycomp), 2)
        self.assertEqual(lbl(m.that), 3)
        self.assertEqual(lbl(self.long1), 4)
        self.assertEqual(lbl(m.myblock), 5)
        self.assertEqual(lbl(m.myblock.mystreet), 6)
        self.assertEqual(lbl(self.thecopy), 7)

    def test_numericlabeler(self):
        m = self.m
        lbl = NumericLabeler('x')
        self.assertEqual(lbl(m.mycomp), 'x1')
        self.assertEqual(lbl(m.mycomp), 'x2')
        self.assertEqual(lbl(m.that), 'x3')
        self.assertEqual(lbl(self.long1), 'x4')
        self.assertEqual(lbl(m.myblock), 'x5')
        self.assertEqual(lbl(m.myblock.mystreet), 'x6')
        self.assertEqual(lbl(self.thecopy), 'x7')

        lbl = NumericLabeler('xyz')
        self.assertEqual(lbl(m.mycomp), 'xyz1')
        self.assertEqual(lbl(m.mycomp), 'xyz2')
        self.assertEqual(lbl(m.that), 'xyz3')
        self.assertEqual(lbl(self.long1), 'xyz4')
        self.assertEqual(lbl(m.myblock), 'xyz5')
        self.assertEqual(lbl(m.myblock.mystreet), 'xyz6')
        self.assertEqual(lbl(self.thecopy), 'xyz7')

    def test_cnamelabeler(self):
        m = self.m
        lbl = CNameLabeler()
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.that), 'that')
        self.assertEqual(lbl(self.long1), 'myverylongcomponentname')
        self.assertEqual(lbl(m.myblock), 'myblock')
        self.assertEqual(lbl(m.myblock.mystreet), 'myblock.mystreet')
        self.assertEqual(lbl(self.thecopy), 'myblock.mystreet')
        self.assertEqual(lbl(m.ind[3]), 'ind[3]')
        self.assertEqual(lbl(m.ind[10]), 'ind[10]')
        self.assertEqual(lbl(m.ind[1]), 'ind[1]')

    def test_textlabeler(self):
        m = self.m
        lbl = TextLabeler()
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.that), 'that')
        self.assertEqual(lbl(self.long1), 'myverylongcomponentname')
        self.assertEqual(lbl(m.myblock), 'myblock')
        self.assertEqual(lbl(m.myblock.mystreet), 'myblock_mystreet')
        self.assertEqual(lbl(self.thecopy), 'myblock_mystreet')
        self.assertEqual(lbl(m.ind[3]), 'ind(3)')
        self.assertEqual(lbl(m.ind[10]), 'ind(10)')
        self.assertEqual(lbl(m.ind[1]), 'ind(1)')

    def test_alphanumerictextlabeler(self):
        m = self.m
        lbl = AlphaNumericTextLabeler()
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.that), 'that')
        self.assertEqual(lbl(self.long1), 'myverylongcomponentname')
        self.assertEqual(lbl(m.myblock), 'myblock')
        self.assertEqual(lbl(m.myblock.mystreet), 'myblock_mystreet')
        self.assertEqual(lbl(self.thecopy), 'myblock_mystreet')
        self.assertEqual(lbl(m.ind[3]), 'ind_3_')
        self.assertEqual(lbl(m.ind[10]), 'ind_10_')
        self.assertEqual(lbl(m.ind[1]), 'ind_1_')

    def test_namelabeler(self):
        m = self.m
        lbl = NameLabeler()
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.that), 'that')
        self.assertEqual(lbl(self.long1), 'myverylongcomponentname')
        self.assertEqual(lbl(m.myblock), 'myblock')
        self.assertEqual(lbl(m.myblock.mystreet), 'myblock.mystreet')
        self.assertEqual(lbl(self.thecopy), 'myblock.mystreet')
        self.assertEqual(lbl(m.ind[3]), 'ind[3]')
        self.assertEqual(lbl(m.ind[10]), 'ind[10]')
        self.assertEqual(lbl(m.ind[1]), 'ind[1]')

    def test_cuidlabeler(self):
        m = self.m
        lbl = CuidLabeler()
        self.assertEqual(lbl(m.mycomp), ComponentUID(m.mycomp))
        self.assertEqual(lbl(m.mycomp), ComponentUID(m.mycomp))
        self.assertEqual(lbl(m.that), ComponentUID(m.that))
        self.assertEqual(lbl(self.long1), ComponentUID(self.long1))
        self.assertEqual(lbl(m.myblock), ComponentUID(m.myblock))
        self.assertEqual(lbl(m.myblock.mystreet), ComponentUID(m.myblock.mystreet))
        self.assertEqual(lbl(self.thecopy), ComponentUID(self.thecopy))
        self.assertEqual(lbl(m.ind[3]), ComponentUID(m.ind[3]))
        self.assertEqual(lbl(m.ind[10]), ComponentUID(m.ind[10]))
        self.assertEqual(lbl(m.ind[1]), ComponentUID(m.ind[1]))

    def test_case_insensitive_shortnamelabeler(self):
        m = self.m
        lbl = ShortNameLabeler(20, '_', caseInsensitive=True)
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.that), 'that')
        self.assertEqual(lbl(self.long1), 'longcomponentname_1_')
        self.assertEqual(lbl(self.long2), 'nentnamerighthere_2_')
        self.assertEqual(lbl(self.long3), 'ngonebutdifferent_3_')
        self.assertEqual(lbl(self.long4), 'longcomponentname_4_')
        self.assertEqual(lbl(self.long5), 'gcomponentname_1__5_')
        self.assertEqual(lbl(m.myblock), 'myblock')
        self.assertEqual(lbl(m.myblock.mystreet), 'myblock_mystreet')
        self.assertEqual(lbl(m.ind[3]), 'ind_3_')
        self.assertEqual(lbl(m.ind[10]), 'ind_10_')
        self.assertEqual(lbl(m.ind[1]), 'ind_1_')

        # Test name collision
        self.assertEqual(lbl(m.mycomp), 'mycomp_6_')
        self.assertEqual(lbl(self.thecopy), 'myblock_mystreet_7_')
        self.assertEqual(lbl(m.MyComp), 'MyComp_8_')

    def test_shortnamelabeler_prefix(self):
        m = self.m
        lbl = ShortNameLabeler(20, '_', prefix='s_', caseInsensitive=True)
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.that), 'that')
        self.assertEqual(lbl(self.long1), 's_ngcomponentname_1_')
        self.assertEqual(lbl(self.long2), 's_ntnamerighthere_2_')
        self.assertEqual(lbl(self.long3), 's_onebutdifferent_3_')
        self.assertEqual(lbl(self.long4), 's_ngcomponentname_4_')
        self.assertEqual(lbl(self.long5), 'longcomponentname_1_')
        self.assertEqual(lbl(m.myblock), 'myblock')
        self.assertEqual(lbl(m.myblock.mystreet), 'myblock_mystreet')
        self.assertEqual(lbl(m.ind[3]), 'ind_3_')
        self.assertEqual(lbl(m.ind[10]), 'ind_10_')
        self.assertEqual(lbl(m.ind[1]), 'ind_1_')

        # Test name collision
        self.assertEqual(lbl(m.mycomp), 's_mycomp_5_')
        self.assertEqual(lbl(self.thecopy), 's_yblock_mystreet_6_')
        self.assertEqual(lbl(m.MyComp), 's_MyComp_7_')

    def test_case_sensitive_shortnamelabeler(self):
        m = self.m
        lbl = ShortNameLabeler(20, '_')
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(m.that), 'that')
        self.assertEqual(lbl(self.long1), 'longcomponentname_1_')
        self.assertEqual(lbl(self.long2), 'nentnamerighthere_2_')
        self.assertEqual(lbl(self.long3), 'ngonebutdifferent_3_')
        self.assertEqual(lbl(self.long4), 'longcomponentname_4_')
        self.assertEqual(lbl(self.long5), 'gcomponentname_1__5_')
        self.assertEqual(lbl(m.myblock), 'myblock')
        self.assertEqual(lbl(m.myblock.mystreet), 'myblock_mystreet')
        self.assertEqual(lbl(m.ind[3]), 'ind_3_')
        self.assertEqual(lbl(m.ind[10]), 'ind_10_')
        self.assertEqual(lbl(m.ind[1]), 'ind_1_')

        # Test name collision
        self.assertEqual(lbl(m.mycomp), 'mycomp')
        self.assertEqual(lbl(self.thecopy), 'myblock_mystreet')
        self.assertEqual(lbl(m.MyComp), 'MyComp')

    def test_case_shortnamelabeler_overflow(self):
        m = self.m
        lbl = ShortNameLabeler(4, '_', caseInsensitive=True)
        for i in range(9):
            self.assertEqual(lbl(m.mycomp), 'p_%d_' % (i+1))
        with self.assertRaisesRegexp(RuntimeError, "Too many identifiers"):
            lbl(m.mycomp)

    def test_shortnamelabeler_legal_regex(self):
        m = ConcreteModel()
        lbl = ShortNameLabeler(
            60, suffix='_', prefix='s_', legalRegex='^[a-zA-Z]')

        m.legal_var = Var()
        self.assertEqual(lbl(m.legal_var), 'legal_var')

        m._illegal_var = Var()
        self.assertEqual(lbl(m._illegal_var), 's__illegal_var_1_')


if __name__ == "__main__":
    unittest.main()
