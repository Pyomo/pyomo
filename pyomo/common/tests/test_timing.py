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

import pyomo.common.unittest as unittest
from pyomo.common.tee import capture_output

import gc
from io import StringIO
from itertools import zip_longest
import logging
import sys
import time

from pyomo.common.log import LoggingIntercept
from pyomo.common.timing import (
    ConstructionTimer,
    TransformationTimer,
    report_timing,
    TicTocTimer,
    HierarchicalTimer,
)
from pyomo.environ import (
    AbstractModel,
    ConcreteModel,
    RangeSet,
    Var,
    Any,
    TransformationFactory,
)
from pyomo.core.base.var import _VarData


class _pseudo_component(Var):
    def getname(*args, **kwds):
        raise RuntimeError("fail")


class TestTiming(unittest.TestCase):
    def setUp(self):
        self.reenable_gc = gc.isenabled()
        gc.disable()

    def tearDown(self):
        if self.reenable_gc:
            gc.enable()
            gc.collect()

    def test_raw_construction_timer(self):
        a = ConstructionTimer(None)
        self.assertRegex(
            str(a),
            r"ConstructionTimer object for NoneType \(unknown\); "
            r"[0-9\.]+ elapsed seconds",
        )
        v = Var()
        v.construct()
        a = ConstructionTimer(_VarData(v))
        self.assertRegex(
            str(a),
            r"ConstructionTimer object for Var ScalarVar\[NOTSET\]; "
            r"[0-9\.]+ elapsed seconds",
        )

    def test_raw_transformation_timer(self):
        a = TransformationTimer(None)
        self.assertRegex(
            str(a), r"TransformationTimer object for NoneType; [0-9\.]+ elapsed seconds"
        )

        v = _pseudo_component()
        a = ConstructionTimer(v)
        self.assertIn("ConstructionTimer object for Var (unknown); ", str(a))

    def test_raw_transformation_timer(self):
        a = TransformationTimer(None, 'fwd')
        self.assertIn("TransformationTimer object for NoneType (fwd); ", str(a))

        a = TransformationTimer(None)
        self.assertIn("TransformationTimer object for NoneType; ", str(a))

    def test_report_timing(self):
        ref = r"""
           (0(\.\d+)?) seconds to construct Block ConcreteModel; 1 index total
           (0(\.\d+)?) seconds to construct RangeSet FiniteScalarRangeSet; 1 index total
           (0(\.\d+)?) seconds to construct Var x; 2 indices total
           (0(\.\d+)?) seconds to construct Var y; 0 indices total
           (0(\.\d+)?) seconds to construct Suffix Suffix
           (0(\.\d+)?) seconds to apply Transformation RelaxIntegerVars \(in-place\)
           """.strip()

        xfrm = TransformationFactory('core.relax_integer_vars')

        try:
            with capture_output() as out:
                report_timing()
                m = ConcreteModel()
                m.r = RangeSet(2)
                m.x = Var(m.r)
                m.y = Var(Any, dense=False)
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
            m.y = Var(Any, dense=False)
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
            m.y = Var(Any, dense=False)
            xfrm.apply_to(m)
            result = os.getvalue().strip()
            self.maxDiff = None
            for l, r in zip(result.splitlines(), ref.splitlines()):
                self.assertRegex(str(l.strip()), str(r.strip()))
            self.assertEqual(buf.getvalue().strip(), "")

    def test_report_timing_context_manager(self):
        ref = r"""
           (0(\.\d+)?) seconds to construct Var x; 2 indices total
           (0(\.\d+)?) seconds to construct Var y; 0 indices total
           (0(\.\d+)?) seconds to construct Suffix Suffix
           (0(\.\d+)?) seconds to apply Transformation RelaxIntegerVars \(in-place\)
           """.strip()

        xfrm = TransformationFactory('core.relax_integer_vars')

        model = AbstractModel()
        model.r = RangeSet(2)
        model.x = Var(model.r)
        model.y = Var(Any, dense=False)

        OS = StringIO()

        with report_timing(False):
            with report_timing(OS):
                with report_timing(False):
                    # Active reporting is False: nothing should be emitted
                    with capture_output() as OUT:
                        m = model.create_instance()
                        xfrm.apply_to(m)
                    self.assertEqual(OUT.getvalue(), "")
                    self.assertEqual(OS.getvalue(), "")
                # Active reporting: we should log the timing
                with capture_output() as OUT:
                    m = model.create_instance()
                    xfrm.apply_to(m)
                self.assertEqual(OUT.getvalue(), "")
                result = OS.getvalue().strip()
                self.maxDiff = None
                for l, r in zip_longest(result.splitlines(), ref.splitlines()):
                    self.assertRegex(str(l.strip()), str(r.strip()))
            # Active reporting is False: the previous log should not have changed
            with capture_output() as OUT:
                m = model.create_instance()
                xfrm.apply_to(m)
            self.assertEqual(OUT.getvalue(), "")
            self.assertEqual(result, OS.getvalue().strip())

    def test_TicTocTimer_tictoc(self):
        SLEEP = 0.1
        RES = 0.02  # resolution (seconds): 1/5 the sleep

        # Note: pypy on GHA occasionally has timing
        # differences of >0.04s
        if 'pypy_version_info' in dir(sys):
            RES *= 2.5
        # Note: previously, OSX on GHA also had significantly nosier tests
        # if sys.platform == 'darwin':
        #     RES *= 2

        abs_time = time.perf_counter()
        timer = TicTocTimer()

        time.sleep(SLEEP)

        with capture_output() as out:
            start_time = time.perf_counter()
            timer.tic(None)
        self.assertEqual(out.getvalue(), '')

        with capture_output() as out:
            start_time = time.perf_counter()
            timer.tic()
        self.assertRegex(
            out.getvalue(), r'\[    [.0-9]+\] Resetting the tic/toc delta timer'
        )

        time.sleep(SLEEP)

        with capture_output() as out:
            ref = time.perf_counter()
            delta = timer.toc()
        self.assertAlmostEqual(ref - start_time, delta, delta=RES)
        self.assertRegex(
            out.getvalue(), r'\[\+   [.0-9]+\] .* in test_TicTocTimer_tictoc'
        )
        with capture_output() as out:
            # entering / leaving the context manager can take non-trivial
            # time on some platforms (up to 0.03 on Windows / Python 3.10)
            self.assertAlmostEqual(
                time.perf_counter() - ref, timer.toc(None), delta=RES
            )
        self.assertEqual(out.getvalue(), '')

        with capture_output() as out:
            ref = time.perf_counter()
            total = timer.toc(delta=False)
        self.assertAlmostEqual(ref - start_time, total, delta=RES)
        self.assertRegex(
            out.getvalue(), r'\[    [.0-9]+\] .* in test_TicTocTimer_tictoc'
        )

        ref *= -1
        time.sleep(SLEEP)

        ref += time.perf_counter()
        timer.stop()
        cumul_stop1 = timer.toc(None)
        self.assertAlmostEqual(ref, cumul_stop1, delta=RES)
        with self.assertRaisesRegex(
            RuntimeError, 'Stopping a TicTocTimer that was already stopped'
        ):
            timer.stop()
        time.sleep(SLEEP)
        cumul_stop2 = timer.toc(None)
        self.assertEqual(cumul_stop1, cumul_stop2)

        ref -= time.perf_counter()
        timer.start()
        time.sleep(SLEEP)

        with capture_output() as out:
            ref += time.perf_counter()
            timer.stop()
            delta = timer.toc()
        self.assertAlmostEqual(ref, delta, delta=RES)
        self.assertRegex(
            out.getvalue(), r'\[    [.0-9]+\|   1\] .* in test_TicTocTimer_tictoc'
        )
        with capture_output() as out:
            # Note that delta is ignored if the timer is a cumulative timer
            total = timer.toc(delta=False)
        self.assertAlmostEqual(delta, total, delta=RES)
        self.assertRegex(
            out.getvalue(), r'\[    [.0-9]+\|   1\] .* in test_TicTocTimer_tictoc'
        )

    def test_TicTocTimer_context_manager(self):
        SLEEP = 0.1
        RES = 0.05  # resolution (seconds): 1/2 the sleep

        abs_time = time.perf_counter()
        with TicTocTimer() as timer:
            time.sleep(SLEEP)
        exclude = -time.perf_counter()
        time.sleep(SLEEP)
        exclude += time.perf_counter()
        with timer:
            time.sleep(SLEEP)
        abs_time = time.perf_counter() - abs_time
        self.assertGreater(abs_time, SLEEP * 3 - RES / 10)
        self.assertAlmostEqual(timer.toc(None), abs_time - exclude, delta=RES)

    def test_TicTocTimer_logger(self):
        # test specifying a logger to the timer constructor disables all
        # output to stdout
        timer = TicTocTimer(logger=logging.getLogger(__name__))
        with capture_output() as OUT:
            with LoggingIntercept(level=logging.INFO) as LOG:
                timer.toc("msg1")
            self.assertRegex(LOG.getvalue(), r"^\[[^\]]+\] msg1$")
            with LoggingIntercept(level=logging.INFO) as LOG:
                timer.toc("msg2", level=logging.DEBUG)
            self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(OUT.getvalue(), "")

        with capture_output() as OUT:
            with LoggingIntercept(level=logging.DEBUG) as LOG:
                timer.toc("msg1")
                timer.toc("msg2", level=logging.DEBUG)
        self.assertEqual(OUT.getvalue(), "")
        self.assertRegex(LOG.getvalue(), r"^\[[^\]]+\] msg1\n\[[^\]]+\] msg2$")

        # test specifying a logger to toc() for a general timer disables
        # output to stdout
        timer = TicTocTimer()
        with capture_output() as OUT:
            with LoggingIntercept(level=logging.INFO) as LOG:
                timer.toc("msg1", logger=logging.getLogger(__name__))
        self.assertRegex(LOG.getvalue(), r"^\[[^\]]+\] msg1$")
        self.assertEqual(OUT.getvalue(), "")

    def test_TicTocTimer_deprecated(self):
        timer = TicTocTimer()
        with LoggingIntercept() as LOG, capture_output() as out:
            timer.tic("msg", None, None)
        self.assertEqual(out.getvalue(), "")
        self.assertRegex(
            LOG.getvalue().replace('\n', ' ').strip(),
            r"DEPRECATED: tic\(\): 'ostream' and 'logger' should be specified "
            r"as keyword arguments( +\([^\)]+\)){2}",
        )

        with LoggingIntercept() as LOG, capture_output() as out:
            timer.toc("msg", True, None, None)
        self.assertEqual(out.getvalue(), "")
        self.assertRegex(
            LOG.getvalue().replace('\n', ' ').strip(),
            r"DEPRECATED: toc\(\): 'delta', 'ostream', and 'logger' should be "
            r"specified as keyword arguments( +\([^\)]+\)){2}",
        )

        timer = TicTocTimer()
        with LoggingIntercept() as LOG, capture_output() as out:
            timer.tic("msg %s, %s", None, None)
        self.assertIn('msg None, None', out.getvalue())
        self.assertEqual(LOG.getvalue(), "")

        with LoggingIntercept() as LOG, capture_output() as out:
            timer.toc("msg %s, %s, %s", True, None, None)
        self.assertIn('msg True, None, None', out.getvalue())
        self.assertEqual(LOG.getvalue(), "")

    def test_HierarchicalTimer(self):
        RES = 0.01  # resolution (seconds)

        timer = HierarchicalTimer()
        start_time = time.perf_counter()
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
        end_time = time.perf_counter()
        timer.stop('all')
        ref = """Identifier        ncalls   cumtime   percall      %
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
            end_time - start_time, timer.get_total_time('all'), delta=RES
        )
        self.assertEqual(100.0, timer.get_relative_percent_time('all'))
        self.assertTrue(100.0 > timer.get_relative_percent_time('all.a'))
        self.assertTrue(50.0 < timer.get_relative_percent_time('all.a'))

    def test_HierarchicalTimer_longNames(self):
        RES = 0.01  # resolution (seconds)

        timer = HierarchicalTimer()
        start_time = time.perf_counter()
        timer.start('all' * 25)
        time.sleep(0.02)
        for i in range(10):
            timer.start('a' * 75)
            time.sleep(0.01)
            for j in range(5):
                timer.start('aa' * 20)
                time.sleep(0.001)
                timer.stop('aa' * 20)
            timer.start('ab' * 20)
            timer.stop('ab' * 20)
            timer.stop('a' * 75)
        end_time = time.perf_counter()
        timer.stop('all' * 25)
        ref = (
            """Identifier%s   ncalls   cumtime   percall      %%
%s------------------------------------
%s%s        1     [0-9.]+ +[0-9.]+ +100.0
    %s------------------------------------
    %s%s       10     [0-9.]+ +[0-9.]+ +[0-9.]+
        %s------------------------------------
        %s%s       50     [0-9.]+ +[0-9.]+ +[0-9.]+
        %s%s       10     [0-9.]+ +[0-9.]+ +[0-9.]+
        other%s      n/a     [0-9.]+ +n/a +[0-9.]+
        %s====================================
    other%s      n/a     [0-9.]+ +n/a +[0-9.]+
    %s====================================
%s====================================
"""
            % (
                ' ' * 69,
                '-' * 79,
                'all' * 25,
                ' ' * 4,
                '-' * 75,
                'a' * 75,
                '',
                '-' * 71,
                'aa' * 20,
                ' ' * 31,
                'ab' * 20,
                ' ' * 31,
                ' ' * 66,
                '=' * 71,
                ' ' * 70,
                '=' * 75,
                '=' * 79,
            )
        ).splitlines()
        for l, r in zip(str(timer).splitlines(), ref):
            self.assertRegex(l, r)

    def test_clear_except_base_timer(self):
        timer = HierarchicalTimer()
        timer.start("a")
        timer.start("b")
        timer.stop("b")
        timer.stop("a")
        timer.start("c")
        timer.stop("c")
        timer.start("d")
        timer.stop("d")
        timer.clear_except("b", "c")
        key_set = set(timer.timers.keys())
        self.assertEqual(key_set, {"c"})

    def test_clear_except_subtimer(self):
        # Testing this method on "sub-timers" exercises different code
        # as while the base timer is a HierarchicalTimer, the sub-timers
        # are _HierarchicalHelpers
        timer = HierarchicalTimer()
        timer.start("root")
        timer.start("a")
        timer.start("b")
        timer.stop("b")
        timer.stop("a")
        timer.start("c")
        timer.stop("c")
        timer.start("d")
        timer.stop("d")
        timer.stop("root")
        root = timer.timers["root"]
        root.clear_except("b", "c")
        key_set = set(root.timers.keys())
        self.assertEqual(key_set, {"c"})


class TestFlattenHierarchicalTimer(unittest.TestCase):
    #
    # The following methods create some hierarchical timers, then
    # hand-code the total time of each timer in the data structure.
    # This is so we can assert that total_time fields of flattened
    # timers are computed correctly without relying on the actual
    # time spent.
    #
    def make_singleton_timer(self):
        timer = HierarchicalTimer()
        timer.start("root")
        timer.stop("root")
        timer.timers["root"].total_time = 5.0
        return timer

    def make_flat_timer(self):
        timer = HierarchicalTimer()
        timer.start("root")
        timer.start("a")
        timer.stop("a")
        timer.start("b")
        timer.stop("b")
        timer.stop("root")
        timer.timers["root"].total_time = 5.0
        timer.timers["root"].timers["a"].total_time = 1.0
        timer.timers["root"].timers["b"].total_time = 2.5
        return timer

    def make_timer_depth_2_one_child(self):
        timer = HierarchicalTimer()
        timer.start("root")
        timer.start("a")
        timer.start("b")
        timer.stop("b")
        timer.start("c")
        timer.stop("c")
        timer.stop("a")
        timer.stop("root")
        timer.timers["root"].total_time = 5.0
        timer.timers["root"].timers["a"].total_time = 4.0
        timer.timers["root"].timers["a"].timers["b"].total_time = 1.1
        timer.timers["root"].timers["a"].timers["c"].total_time = 2.2
        return timer

    def make_timer_depth_2_with_name_collision(self):
        timer = HierarchicalTimer()
        timer.start("root")
        timer.start("a")
        timer.start("b")
        timer.stop("b")
        timer.start("c")
        timer.stop("c")
        timer.stop("a")
        timer.start("b")
        timer.stop("b")
        timer.stop("root")
        timer.timers["root"].total_time = 5.0
        timer.timers["root"].timers["a"].total_time = 4.0
        timer.timers["root"].timers["a"].timers["b"].total_time = 1.1
        timer.timers["root"].timers["a"].timers["c"].total_time = 2.2
        timer.timers["root"].timers["b"].total_time = 0.11
        return timer

    def make_timer_depth_2_two_children(self):
        timer = HierarchicalTimer()
        timer.start("root")
        timer.start("a")
        timer.start("b")
        timer.stop("b")
        timer.start("c")
        timer.stop("c")
        timer.stop("a")
        timer.start("b")
        timer.start("c")
        timer.stop("c")
        timer.start("d")
        timer.stop("d")
        timer.stop("b")
        timer.stop("root")
        timer.timers["root"].total_time = 5.0
        timer.timers["root"].timers["a"].total_time = 4.0
        timer.timers["root"].timers["a"].timers["b"].total_time = 1.1
        timer.timers["root"].timers["a"].timers["c"].total_time = 2.2
        timer.timers["root"].timers["b"].total_time = 0.88
        timer.timers["root"].timers["b"].timers["c"].total_time = 0.07
        timer.timers["root"].timers["b"].timers["d"].total_time = 0.05
        return timer

    def make_timer_depth_4(self):
        timer = HierarchicalTimer()
        timer.start("root")
        timer.start("a")
        timer.start("b")
        timer.stop("b")
        timer.start("c")
        timer.start("d")
        timer.start("e")
        timer.stop("e")
        timer.stop("d")
        timer.stop("c")
        timer.stop("a")
        timer.start("b")
        timer.start("c")
        timer.start("e")
        timer.stop("e")
        timer.stop("c")
        timer.start("d")
        timer.stop("d")
        timer.stop("b")
        timer.stop("root")
        timer.timers["root"].total_time = 5.0
        timer.timers["root"].timers["a"].total_time = 4.0
        timer.timers["root"].timers["a"].timers["b"].total_time = 1.1
        timer.timers["root"].timers["a"].timers["c"].total_time = 2.2
        timer.timers["root"].timers["a"].timers["c"].timers["d"].total_time = 0.9
        timer.timers["root"].timers["a"].timers["c"].timers["d"].timers[
            "e"
        ].total_time = 0.6
        timer.timers["root"].timers["b"].total_time = 0.88
        timer.timers["root"].timers["b"].timers["c"].total_time = 0.07
        timer.timers["root"].timers["b"].timers["c"].timers["e"].total_time = 0.04
        timer.timers["root"].timers["b"].timers["d"].total_time = 0.05
        return timer

    def make_timer_depth_4_same_name(self):
        timer = HierarchicalTimer()
        timer.start("root")
        timer.start("a")
        timer.start("a")
        timer.start("a")
        timer.start("a")
        timer.stop("a")
        timer.stop("a")
        timer.stop("a")
        timer.stop("a")
        timer.stop("root")
        timer.timers["root"].total_time = 5.0
        timer.timers["root"].timers["a"].total_time = 1.0
        timer.timers["root"].timers["a"].timers["a"].total_time = 0.1
        timer.timers["root"].timers["a"].timers["a"].timers["a"].total_time = 0.01
        timer.timers["root"].timers["a"].timers["a"].timers["a"].timers[
            "a"
        ].total_time = 0.001
        return timer

    def test_singleton(self):
        timer = self.make_singleton_timer()
        root = timer.timers["root"]
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)

    def test_already_flat(self):
        timer = self.make_flat_timer()
        root = timer.timers["root"]
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers["a"].total_time, 1.0)
        self.assertAlmostEqual(root.timers["b"].total_time, 2.5)

    def test_depth_2_one_child(self):
        timer = self.make_timer_depth_2_one_child()
        root = timer.timers["root"]
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers["a"].total_time, 0.7)
        self.assertAlmostEqual(root.timers["b"].total_time, 1.1)
        self.assertAlmostEqual(root.timers["c"].total_time, 2.2)

    def test_timer_depth_2_with_name_collision(self):
        timer = self.make_timer_depth_2_with_name_collision()
        root = timer.timers["root"]
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers["a"].total_time, 0.700)
        self.assertAlmostEqual(root.timers["b"].total_time, 1.210)
        self.assertAlmostEqual(root.timers["c"].total_time, 2.200)

    def test_timer_depth_2_two_children(self):
        timer = self.make_timer_depth_2_two_children()
        root = timer.timers["root"]
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers["a"].total_time, 0.700)
        self.assertAlmostEqual(root.timers["b"].total_time, 1.860)
        self.assertAlmostEqual(root.timers["c"].total_time, 2.270)
        self.assertAlmostEqual(root.timers["d"].total_time, 0.050)

    def test_timer_depth_4(self):
        timer = self.make_timer_depth_4()
        root = timer.timers["root"]
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers["a"].total_time, 0.700)
        self.assertAlmostEqual(root.timers["b"].total_time, 1.860)
        self.assertAlmostEqual(root.timers["c"].total_time, 1.330)
        self.assertAlmostEqual(root.timers["d"].total_time, 0.350)
        self.assertAlmostEqual(root.timers["e"].total_time, 0.640)

    def test_timer_depth_4_same_name(self):
        timer = self.make_timer_depth_4_same_name()
        root = timer.timers["root"]
        root.flatten()
        self.assertAlmostEqual(root.total_time, 5.0)
        self.assertAlmostEqual(root.timers["a"].total_time, 1.0)

    def test_base_timer_depth_3(self):
        # This is depth 2 wrt the root, depth 3 wrt the
        # "base timer"
        timer = self.make_timer_depth_2_two_children()
        timer.flatten()
        self.assertAlmostEqual(timer.timers["root"].total_time, 0.120)
        self.assertAlmostEqual(timer.timers["a"].total_time, 0.700)
        self.assertAlmostEqual(timer.timers["b"].total_time, 1.860)
        self.assertAlmostEqual(timer.timers["c"].total_time, 2.270)
        self.assertAlmostEqual(timer.timers["d"].total_time, 0.050)

    def test_timer_still_active(self):
        timer = HierarchicalTimer()
        timer.start("a")
        timer.stop("a")
        timer.start("b")
        msg = "Cannot flatten.*while any timers are active"
        with self.assertRaisesRegex(RuntimeError, msg):
            timer.flatten()
        timer.stop("b")
