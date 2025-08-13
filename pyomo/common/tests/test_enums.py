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

import enum

import pyomo.common.unittest as unittest

from pyomo.common.enums import ExtendedEnumType, ObjectiveSense, SolverAPIVersion


class ProblemSense(enum.IntEnum, metaclass=ExtendedEnumType):
    __base_enum__ = ObjectiveSense

    unknown = 0


class TestExtendedEnumType(unittest.TestCase):
    def test_members(self):
        self.assertEqual(
            list(ProblemSense),
            [ProblemSense.unknown, ObjectiveSense.minimize, ObjectiveSense.maximize],
        )

    def test_isinstance(self):
        self.assertIsInstance(ProblemSense.unknown, ProblemSense)
        self.assertIsInstance(ProblemSense.minimize, ProblemSense)
        self.assertIsInstance(ProblemSense.maximize, ProblemSense)

        self.assertTrue(ProblemSense.__instancecheck__(ProblemSense.unknown))
        self.assertTrue(ProblemSense.__instancecheck__(ProblemSense.minimize))
        self.assertTrue(ProblemSense.__instancecheck__(ProblemSense.maximize))

    def test_getattr(self):
        self.assertIs(ProblemSense.unknown, ProblemSense.unknown)
        self.assertIs(ProblemSense.minimize, ObjectiveSense.minimize)
        self.assertIs(ProblemSense.maximize, ObjectiveSense.maximize)

    def test_hasattr(self):
        self.assertTrue(hasattr(ProblemSense, 'unknown'))
        self.assertTrue(hasattr(ProblemSense, 'minimize'))
        self.assertTrue(hasattr(ProblemSense, 'maximize'))

    def test_call(self):
        self.assertIs(ProblemSense(0), ProblemSense.unknown)
        self.assertIs(ProblemSense(1), ObjectiveSense.minimize)
        self.assertIs(ProblemSense(-1), ObjectiveSense.maximize)

        self.assertIs(ProblemSense('unknown'), ProblemSense.unknown)
        self.assertIs(ProblemSense('minimize'), ObjectiveSense.minimize)
        self.assertIs(ProblemSense('maximize'), ObjectiveSense.maximize)

        with self.assertRaisesRegex(ValueError, "'foo' is not a valid ProblemSense"):
            ProblemSense('foo')
        with self.assertRaisesRegex(ValueError, "2 is not a valid ProblemSense"):
            ProblemSense(2)

    def test_contains(self):
        self.assertIn(ProblemSense.unknown, ProblemSense)
        self.assertIn(ProblemSense.minimize, ProblemSense)
        self.assertIn(ProblemSense.maximize, ProblemSense)

        self.assertNotIn(ProblemSense.unknown, ObjectiveSense)
        self.assertIn(ProblemSense.minimize, ObjectiveSense)
        self.assertIn(ProblemSense.maximize, ObjectiveSense)


class TestObjectiveSense(unittest.TestCase):
    def test_members(self):
        self.assertEqual(
            list(ObjectiveSense), [ObjectiveSense.minimize, ObjectiveSense.maximize]
        )

    def test_hasattr(self):
        self.assertTrue(hasattr(ProblemSense, 'minimize'))
        self.assertTrue(hasattr(ProblemSense, 'maximize'))

    def test_call(self):
        self.assertIs(ObjectiveSense(1), ObjectiveSense.minimize)
        self.assertIs(ObjectiveSense(-1), ObjectiveSense.maximize)

        self.assertIs(ObjectiveSense('minimize'), ObjectiveSense.minimize)
        self.assertIs(ObjectiveSense('maximize'), ObjectiveSense.maximize)

        with self.assertRaisesRegex(ValueError, "'foo' is not a valid ObjectiveSense"):
            ObjectiveSense('foo')

    def test_str(self):
        self.assertEqual(str(ObjectiveSense.minimize), 'minimize')
        self.assertEqual(str(ObjectiveSense.maximize), 'maximize')


class TestSolverAPIVersion(unittest.TestCase):
    def test_members(self):
        self.assertEqual(
            list(SolverAPIVersion),
            [SolverAPIVersion.V1, SolverAPIVersion.APPSI, SolverAPIVersion.V2],
        )

    def test_call(self):
        self.assertIs(SolverAPIVersion(10), SolverAPIVersion.V1)
        self.assertIs(SolverAPIVersion(15), SolverAPIVersion.APPSI)
        self.assertIs(SolverAPIVersion(20), SolverAPIVersion.V2)

        self.assertIs(SolverAPIVersion('V1'), SolverAPIVersion.V1)
        self.assertIs(SolverAPIVersion('APPSI'), SolverAPIVersion.APPSI)
        self.assertIs(SolverAPIVersion('V2'), SolverAPIVersion.V2)

        with self.assertRaisesRegex(
            ValueError, "'foo' is not a valid SolverAPIVersion"
        ):
            SolverAPIVersion('foo')
