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

import pyomo.common.unittest as unittest

import inspect

from pyomo.common.pyomo_typing import get_overloads_for
from pyomo.environ import Block


class TestTyping(unittest.TestCase):
    def test_get_overloads_for(self):
        func_list = get_overloads_for(Block.__init__)
        self.assertEqual(len(func_list), 1)
        kwds = inspect.getfullargspec(func_list[0]).kwonlyargs
        self.assertEqual(kwds, ['rule', 'concrete', 'dense', 'name', 'doc'])
