#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys

import pyomo.common.unittest as unittest

from pyomo.common.flags import NOTSET, in_testing_environment, building_documentation


class TestModeling(unittest.TestCase):

    def test_NOTSET(self):
        self.assertEqual(str(NOTSET), 'NOTSET')
        self.assertNotIn('sphinx', sys.modules)
        self.assertEqual(repr(NOTSET), 'pyomo.common.flags.NOTSET')
        self.assertIsNone(in_testing_environment.state)

        self.assertTrue(in_testing_environment())
        self.assertFalse(building_documentation())
        try:
            sys.modules['sphinx'] = sys.modules[__name__]
            for i in sorted(sys.modules.items()):
                print(i)
            self.assertTrue(in_testing_environment())
            self.assertTrue(building_documentation())
            self.assertEqual(repr(NOTSET), 'pyomo.common.flags.NOTSET')

            in_testing_environment(False)
            self.assertFalse(in_testing_environment())
            self.assertTrue(building_documentation())
            self.assertEqual(repr(NOTSET), 'NOTSET')
        finally:
            del sys.modules['sphinx']
            in_testing_environment(None)
        self.assertIsNone(in_testing_environment.state)
