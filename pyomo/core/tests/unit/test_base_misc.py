# -*- coding: utf-8 -*- 
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
# Unit Tests for pyomo.base.misc
#
from six import StringIO, iteritems

import pyutilib.th as unittest

from pyomo.core.base.misc import tabular_writer

class TestTabularWriter(unittest.TestCase):
    def test_unicode_table(self):
        # Test that an embedded unicode character does not foul up the
        # table alignment
        os = StringIO()
        data = {1: ("a", 1), (2,3): ("∧", 2)}
        tabular_writer(os, "", iteritems(data), ["s", "val"], lambda k,v: v)
        ref = u"""
Key    : s : val
     1 : a :   1
(2, 3) : ∧ :   2
"""
        self.assertEqual(ref.strip(), os.getvalue().strip())

if __name__ == "__main__":
    unittest.main()
    
