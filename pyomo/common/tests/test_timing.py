#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyutilib.misc.redirect_io import capture_output

from six import StringIO
import sys
import time

from pyomo.common.log import LoggingIntercept
from pyomo.common.timing import (ConstructionTimer, report_timing,
                                 TicTocTimer, HierarchicalTimer)
from pyomo.environ import ConcreteModel, RangeSet, Var, TransformationFactory

class TestTiming(unittest.TestCase):
    def test_raw_construction_timer(self):
        a = ConstructionTimer(None)
        self.assertIn(
            "ConstructionTimer object for NoneType (unknown); ",
            str(a))

    def test_report_timing(self):
        # Create a set to ensure that the global sets have already been
        # constructed (this is an issue until the new set system is
        # merged in and the GlobalSet objects are not automatically
        # created by pyomo.core
        m = ConcreteModel()
        m.x = Var([1,2])

        ref = r"""
           (0(\.\d+)?) seconds to construct Block ConcreteModel; 1 index total
           (0(\.\d+)?) seconds to construct RangeSet FiniteSimpleRangeSet; 1 index total
           (0(\.\d+)?) seconds to construct Var x; 2 indices total
           (0(\.\d+)?) seconds to construct Suffix Suffix; 1 index total
           (0(\.\d+)?) seconds to apply Transformation RelaxIntegerVars \(in-place\)
           """.strip()

        xfrm = TransformationFactory('core.relax_integer_vars')

        try:
            with capture_output() as out:
                report_timing()
                m = ConcreteModel()
                m.r = RangeSet(2)
                m.x = Var(m.r)
                xfrm.apply_to(m)
            result = out.getvalue().strip()
            self.maxDiff = None
            for l, r in zip(result.splitlines(), ref.splitlines()):
                self.assertRegex(str(l.strip()), str(r.strip()))
        finally:
            report_timing(False)

        os = StringIO()
        try:
            report_timing(os)
            m = ConcreteModel()
            m.r = RangeSet(2)
            m.x = Var(m.r)
            xfrm.apply_to(m)
            result = os.getvalue().strip()
            self.maxDiff = None
            for l, r in zip(result.splitlines(), ref.splitlines()):
                self.assertRegex(str(l.strip()), str(r.strip()))
        finally:
            report_timing(False)
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo'):
            m = ConcreteModel()
            m.r = RangeSet(2)
            m.x = Var(m.r)
            xfrm.apply_to(m)
            result = os.getvalue().strip()
            self.maxDiff = None
            for l, r in zip(result.splitlines(), ref.splitlines()):
                self.assertRegex(str(l.strip()), str(r.strip()))
            self.assertEqual(buf.getvalue().strip(), "")

    def test_TicTocTimer_tictoc(self):
        RES = 1e-2 # resolution (seconds)
        timer = TicTocTimer()
        abs_time = time.time()

        time.sleep(0.1)

        with capture_output() as out:
            start_time = time.time()
            timer.tic(None)
        self.assertEqual(out.getvalue(), '')

        with capture_output() as out:
            start_time = time.time()
            timer.tic()
        self.assertRegex(
            out.getvalue(),
            r'\[    [.0-9]+\] Resetting the tic/toc delta timer'
        )

        time.sleep(0.1)

        with capture_output() as out:
            delta = timer.toc()
        self.assertAlmostEqual(time.time() - start_time, delta, delta=RES)
        self.assertAlmostEqual(0, timer.toc(None), delta=RES)
        self.assertRegex(
            out.getvalue(),
            r'\[\+   [.0-9]+\] .* in test_TicTocTimer_tictoc'
        )
        with capture_output() as out:
            total = timer.toc(delta=False)
        self.assertAlmostEqual(time.time() - abs_time, total, delta=RES)
        self.assertRegex(
            out.getvalue(),
            r'\[    [.0-9]+\] .* in test_TicTocTimer_tictoc'
        )

        ref = 0
        ref -= time.time()
        time.sleep(0.1)
        timer.stop()
        ref += time.time()
        cumul_stop1 = timer.toc(None)
        with self.assertRaisesRegex(
                RuntimeError,
                'Stopping a TicTocTimer that was already stopped'):
            timer.stop()
        time.sleep(0.1)
        cumul_stop2 = timer.toc(None)
        self.assertEqual(cumul_stop1, cumul_stop2)
        timer.start()
        ref -= time.time()
        time.sleep(0.1)

        # Note: pypy on GHA frequently has timing differences of >0.05s
        # for the following tests
        if 'pypy_version_info' in dir(sys):
            RES = 6.5e-2

        with capture_output() as out:
            delta = timer.toc()
            timer.stop()
            ref += time.time()
        self.assertAlmostEqual(ref, delta, delta=RES)
        #self.assertAlmostEqual(0, timer.toc(None), delta=RES)
        self.assertRegex(
            out.getvalue(),
            r'\[    [.0-9]+\|   1\] .* in test_TicTocTimer_tictoc'
        )
        with capture_output() as out:
            # Note that delta is ignored if the timer is a cumulative timer
            total = timer.toc(delta=False)
        self.assertAlmostEqual(delta, total, delta=RES)
        self.assertRegex(
            out.getvalue(),
            r'\[    [.0-9]+\|   1\] .* in test_TicTocTimer_tictoc'
        )

    def test_HierarchicalTimer(self):
        RES = 1e-2 # resolution (seconds)

        timer = HierarchicalTimer()
        start_time = time.time()
        timer.start('all')
        time.sleep(0.02)
        for i in range(10):
            timer.start('a')
            time.sleep(0.01)
            for j in range(5):
                timer.start('aa')
                time.sleep(0.001)
                timer.stop('aa')
            timer.start('ab')
            timer.stop('ab')
            timer.stop('a')
        end_time = time.time()
        timer.stop('all')
        ref = \
"""Identifier        ncalls   cumtime   percall      %
---------------------------------------------------
all                    1     [0-9.]+ +[0-9.]+ +100.0
     ----------------------------------------------
     a                10     [0-9.]+ +[0-9.]+ +[0-9.]+
          -----------------------------------------
          aa          50     [0-9.]+ +[0-9.]+ +[0-9.]+
          ab          10     [0-9.]+ +[0-9.]+ +[0-9.]+
          other      n/a     [0-9.]+ +n/a +[0-9.]+
          =========================================
     other           n/a     [0-9.]+ +n/a +[0-9.]+
     ==============================================
===================================================
""".splitlines()
        for l, r in zip(str(timer).splitlines(), ref):
            self.assertRegex(l, r)

        self.assertEqual(1, timer.get_num_calls('all'))
        self.assertAlmostEqual(
            end_time - start_time, timer.get_total_time('all'), delta=RES)
        self.assertEqual(100., timer.get_relative_percent_time('all'))
        self.assertTrue(100. > timer.get_relative_percent_time('all.a'))
        self.assertTrue(50. < timer.get_relative_percent_time('all.a'))
