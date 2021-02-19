#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.gc_manager import PauseGC
import gc

import pyutilib.th as unittest

class TestPauseGC(unittest.TestCase):
    def test_gc_disable(self):
        self.assertTrue(gc.isenabled())
        pgc = PauseGC()
        self.assertFalse(gc.isenabled())
        with PauseGC():
            self.assertFalse(gc.isenabled())
        self.assertFalse(gc.isenabled())
        pgc.close()
        self.assertTrue(gc.isenabled())

        self.assertTrue(gc.isenabled())
        with PauseGC():
            self.assertFalse(gc.isenabled())
            pgc = PauseGC()
            self.assertFalse(gc.isenabled())
            pgc.close()
            self.assertFalse(gc.isenabled())
        self.assertTrue(gc.isenabled())
