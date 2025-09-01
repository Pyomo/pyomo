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

import sys

import pyomo
import pyomo.common.unittest as unittest

from pyomo.common.flags import NOTSET, in_testing_environment, building_documentation


class TestFlags(unittest.TestCase):

    def test_NOTSET(self):
        self.assertTrue(in_testing_environment())
        self.assertFalse(building_documentation())

        self.assertEqual(str(NOTSET), 'NOTSET')
        self.assertFalse(hasattr(pyomo, '__sphinx_build__'))
        self.assertEqual(repr(NOTSET), 'pyomo.common.flags.NOTSET')
        self.assertIsNone(in_testing_environment.state)

        try:
            pyomo.__sphinx_build__ = True
            self.assertTrue(in_testing_environment())
            self.assertTrue(building_documentation())
            self.assertEqual(repr(NOTSET), 'NOTSET')

            in_testing_environment(False)
            self.assertFalse(in_testing_environment())
            self.assertTrue(building_documentation())
            self.assertEqual(repr(NOTSET), 'NOTSET')
        finally:
            del pyomo.__sphinx_build__
            in_testing_environment(None)
        self.assertIsNone(in_testing_environment.state)

    def test_singleton(self):
        # This tests that the type is a "singleton", and that any
        # attempts to construct an instance will return the class
        self.assertIs(NOTSET(), NOTSET)
        self.assertIs(NOTSET(), NOTSET())
