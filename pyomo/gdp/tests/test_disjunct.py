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

from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct

from six import iterkeys

class TestDisjunction(unittest.TestCase):
    def test_empty_disjunction(self):
        m = ConcreteModel()
        m.d = Disjunct()
        m.e = Disjunct()

        m.x1 = Disjunction()
        self.assertEqual(len(m.x1), 0)

        m.x1 = [m.d, m.e]
        self.assertEqual(len(m.x1), 1)
        self.assertEqual(m.x1.disjuncts, [m.d, m.e])

        m.x2 = Disjunction([1,2,3,4])
        self.assertEqual(len(m.x2), 0)

        m.x2[2] = [m.d, m.e]
        self.assertEqual(len(m.x2), 1)
        self.assertEqual(m.x2[2].disjuncts, [m.d, m.e])

    def test_construct_implicit_disjuncts(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.d = Disjunction(expr=[m.x<=0, m.y>=1])
        self.assertEqual(len(m.component_map(Disjunction)), 1)
        self.assertEqual(len(m.component_map(Disjunct)), 1)

        implicit_disjuncts = list(iterkeys(m.component_map(Disjunct)))
        self.assertEqual(implicit_disjuncts[0][:2], "d_")
        disjuncts = m.d.disjuncts
        self.assertEqual(len(disjuncts), 2)
        self.assertIs(disjuncts[0].parent_block(), m)
        self.assertIs(disjuncts[0].constraint[1].body, m.x)
        self.assertIs(disjuncts[1].parent_block(), m)
        self.assertIs(disjuncts[1].constraint[1].body, m.y)

        # Test that the implicit disjuncts get a unique name
        m.add_component('e_disjuncts', Var())
        m.e = Disjunction(expr=[m.y<=0, m.x>=1])
        self.assertEqual(len(m.component_map(Disjunction)), 2)
        self.assertEqual(len(m.component_map(Disjunct)), 2)
        implicit_disjuncts = list(iterkeys(m.component_map(Disjunct)))
        self.assertEqual(implicit_disjuncts[1][:12], "e_disjuncts_")
        disjuncts = m.e.disjuncts
        self.assertEqual(len(disjuncts), 2)
        self.assertIs(disjuncts[0].parent_block(), m)
        self.assertIs(disjuncts[0].constraint[1].body, m.y)
        self.assertIs(disjuncts[1].parent_block(), m)
        self.assertIs(disjuncts[1].constraint[1].body, m.x)
        self.assertEqual(len(disjuncts[0].parent_component().name), 13)
        self.assertEqual(disjuncts[0].name[:12], "e_disjuncts_")

        # Test that the implicit disjuncts can be lists/tuples/generators
        def _gen():
            yield m.y<=4
            yield m.x>=5
        m.f = Disjunction(expr=[
            [ m.y<=0,
              m.x>=1 ],
            ( m.y<=2,
              m.x>=3 ),
            _gen() ])
        self.assertEqual(len(m.component_map(Disjunction)), 3)
        self.assertEqual(len(m.component_map(Disjunct)), 3)
        implicit_disjuncts = list(iterkeys(m.component_map(Disjunct)))
        self.assertEqual(implicit_disjuncts[2][:12], "f_disjuncts")
        disjuncts = m.f.disjuncts
        self.assertEqual(len(disjuncts), 3)
        self.assertIs(disjuncts[0].parent_block(), m)
        self.assertIs(disjuncts[0].constraint[1].body, m.y)
        self.assertEqual(disjuncts[0].constraint[1].upper, 0)
        self.assertIs(disjuncts[0].constraint[2].body, m.x)
        self.assertEqual(disjuncts[0].constraint[2].lower, 1)

        self.assertIs(disjuncts[1].parent_block(), m)
        self.assertIs(disjuncts[1].constraint[1].body, m.y)
        self.assertEqual(disjuncts[1].constraint[1].upper, 2)
        self.assertIs(disjuncts[1].constraint[2].body, m.x)
        self.assertEqual(disjuncts[1].constraint[2].lower, 3)

        self.assertIs(disjuncts[2].parent_block(), m)
        self.assertIs(disjuncts[2].constraint[1].body, m.y)
        self.assertEqual(disjuncts[2].constraint[1].upper, 4)
        self.assertIs(disjuncts[2].constraint[2].body, m.x)
        self.assertEqual(disjuncts[2].constraint[2].lower, 5)

        self.assertEqual(len(disjuncts[0].parent_component().name), 11)
        self.assertEqual(disjuncts[0].name, "f_disjuncts[0]")


class TestDisjunct(unittest.TestCase):
    def test_deactivate(self):
        m = ConcreteModel()
        m.x = Var()
        m.d1 = Disjunct()
        m.d1.constraint = Constraint(expr=m.x<=0)
        m.d = Disjunction(expr=[m.d1, m.x>=1, m.x>=5])
        d2 = m.d.disjuncts[1].parent_component()
        self.assertEqual(len(m.component_map(Disjunction)), 1)
        self.assertEqual(len(m.component_map(Disjunct)), 2)
        self.assertIsNot(m.d1, d2)

        self.assertTrue(m.d1.active)
        self.assertTrue(d2.active)
        self.assertTrue(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[0].deactivate()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[1].deactivate()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertFalse(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertTrue(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        d2.deactivate()
        self.assertFalse(m.d1.active)
        self.assertFalse(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertFalse(m.d.disjuncts[1].active)
        self.assertFalse(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertTrue(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertTrue(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[2].activate()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertFalse(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertTrue(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        d2.activate()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d1.activate()
        self.assertTrue(m.d1.active)
        self.assertTrue(d2.active)
        self.assertTrue(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

    def test_deactivate_without_fixing_indicator(self):
        m = ConcreteModel()
        m.x = Var()
        m.d1 = Disjunct()
        m.d1.constraint = Constraint(expr=m.x<=0)
        m.d = Disjunction(expr=[m.d1, m.x>=1, m.x>=5])
        d2 = m.d.disjuncts[1].parent_component()
        self.assertEqual(len(m.component_map(Disjunction)), 1)
        self.assertEqual(len(m.component_map(Disjunct)), 2)
        self.assertIsNot(m.d1, d2)

        self.assertTrue(m.d1.active)
        self.assertTrue(d2.active)
        self.assertTrue(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[0]._deactivate_without_fixing_indicator()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[1]._deactivate_without_fixing_indicator()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertFalse(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())


if __name__ == '__main__':
    unittest.main()

