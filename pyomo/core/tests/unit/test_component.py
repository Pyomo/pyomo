#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for components
#
from six import StringIO
import pyutilib.th as unittest

from pyomo.common import DeveloperError
import pyomo.core.base._pyomo
from pyomo.environ import (
    ConcreteModel, Component, Block, Var, Set, Param,
)

class TestComponent(unittest.TestCase):

    def test_construct_component_throws_exception(self):
        with self.assertRaisesRegexp(
                DeveloperError,
                "Must specify a component type for class Component"):
            Component()

    def test_getname(self):
        m = ConcreteModel()
        m.b = Block([1,2])
        m.b[2].c = Var([1,2],[3,4])

        self.assertEqual(m.getname(fully_qualified=True), "unknown")
        self.assertEqual(m.b.getname(fully_qualified=True), "b")
        self.assertEqual(m.b[1].getname(fully_qualified=True), "b[1]")
        self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=True),
                         "b[2].c[2,4]")
        self.assertEqual(m.getname(fully_qualified=False), "unknown")
        self.assertEqual(m.b.getname(fully_qualified=False), "b")
        self.assertEqual(m.b[1].getname(fully_qualified=False), "b[1]")
        self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=False),
                         "c[2,4]")

        cache = {}
        self.assertEqual(
            m.b[2].c[2,4].getname(fully_qualified=True, name_buffer=cache),
            "b[2].c[2,4]")
        self.assertEqual(len(cache), 8)
        self.assertIn(id(m.b[2].c[2,4]), cache)
        self.assertIn(id(m.b[2].c[1,3]), cache)
        self.assertIn(id(m.b[2].c), cache)
        self.assertIn(id(m.b[2]), cache)
        self.assertIn(id(m.b[1]), cache)
        self.assertIn(id(m.b), cache)
        self.assertNotIn(id(m), cache)
        self.assertEqual(
            m.b[2].c[1,3].getname(fully_qualified=True, name_buffer=cache),
            "b[2].c[1,3]")

        m.b[2]._component = None
        self.assertEqual(m.b[2].getname(fully_qualified=True),
                         "[Unattached _BlockData]")
        # I think that getname() should do this:
        #self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=True),
        #                 "[Unattached _BlockData].c[2,4]")
        # but it doesn't match current behavior.  I will file a PEP to
        # propose changing the behavior later and proceed to test
        # current behavior.
        self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=True),
                         "c[2,4]")

        self.assertEqual(m.b[2].getname(fully_qualified=False),
                         "[Unattached _BlockData]")
        self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=False),
                         "c[2,4]")

        # Cached names still work...
        self.assertEqual(
            m.b[2].getname(fully_qualified=True, name_buffer=cache),
            "b[2]")
        self.assertEqual(
            m.b[2].c[1,3].getname(fully_qualified=True, name_buffer=cache),
            "b[2].c[1,3]")

    def test_component_data_pprint(self):
        m = ConcreteModel()
        m.a = Set(initialize=[1, 2, 3], ordered=True)
        m.x = Var(m.a)
        stream = StringIO()
        m.x[2].pprint(ostream=stream)
        correct_s = '{Member of x} : Size=3, Index=a\n    ' \
                    'Key : Lower : Value : Upper : Fixed : Stale : Domain\n      ' \
                    '2 :  None :  None :  None : False :  True :  Reals\n'
        self.assertEqual(correct_s, stream.getvalue())

    def test_is_reference(self):
        m = ConcreteModel()
        class _NotSpecified(object):
            pass
        m.comp = Component(ctype=_NotSpecified)
        self.assertFalse(m.comp.is_reference())

class TestEnviron(unittest.TestCase):

    def test_components(self):
        self.assertGreaterEqual(
            set(x[0] for x in pyomo.core.base._pyomo.model_components()),
            set(['Set', 'Param', 'Var', 'Objective', 'Constraint'])
        )

    def test_sets(self):
        self.assertGreaterEqual(
            set(x[0] for x in pyomo.core.base._pyomo.predefined_sets()),
            set(['Reals', 'Integers', 'Boolean'])
        )

if __name__ == "__main__":
    unittest.main()
