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

from pyomo.common.gc_manager import PauseGC
import gc

import pyomo.common.unittest as unittest


class TestPauseGC(unittest.TestCase):
    def test_gc_disable(self):
        self.assertTrue(gc.isenabled())
        pgc = PauseGC()
        self.assertTrue(gc.isenabled())
        # pgc hasn't been entered.  Nothing to do
        pgc.close()
        self.assertTrue(gc.isenabled())

        with pgc:
            self.assertFalse(gc.isenabled())
        self.assertTrue(gc.isenabled())

    def test_gc_early_close(self):
        pgc = PauseGC()
        with pgc:
            self.assertFalse(gc.isenabled())
            pgc.close()
            self.assertTrue(gc.isenabled())
        self.assertTrue(gc.isenabled())

    def test_gc_nested(self):
        pgc = PauseGC()
        with pgc:
            self.assertFalse(gc.isenabled())
            with PauseGC():
                self.assertFalse(gc.isenabled())
            self.assertFalse(gc.isenabled())
        self.assertTrue(gc.isenabled())

    def test_gc_errors(self):
        pgc = PauseGC()
        self.assertTrue(gc.isenabled())

        with pgc:
            with self.assertRaisesRegex(
                RuntimeError,
                "Entering PauseGC context manager that was already entered",
            ):
                with pgc:
                    pass
            self.assertFalse(gc.isenabled())
        self.assertTrue(gc.isenabled())

        with pgc:
            self.assertFalse(gc.isenabled())
            with PauseGC():
                self.assertFalse(gc.isenabled())
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Exiting PauseGC context manager out of order: there "
                    "are other active PauseGC context managers that were "
                    "entered after this context manager and have not yet "
                    "been exited.",
                ):
                    pgc.close()
            self.assertFalse(gc.isenabled())
        self.assertTrue(gc.isenabled())
