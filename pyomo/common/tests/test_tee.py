# -*- coding: utf-8 -*-
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

import gc
import itertools
import logging
import os
import platform
import time
import sys

from io import StringIO, BytesIO

from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee


class timestamper:
    """A 'TextIO'-like object that records the time when data was written to
    the stream."""

    def __init__(self):
        self.buf = []
        self.error = ""

    def write(self, data):
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            self.buf.append((time.time(), float(line)))

    def writelines(self, data):
        for line in data:
            self.write(line.strip())

    def flush(self):
        pass

    def check(self, *bases):
        """Map the recorded times to {0, 1} based on the range of times
        recorded: anything in the first half of the range is mapped to
        0, and anything in the second half is mapped to 1.  This
        "discretizes" the times so that we can reliably compare to
        baselines.

        """

        n = list(itertools.chain(*self.buf))
        mid = (min(n) + max(n)) / 2.0
        result = [tuple(0 if i < mid else 1 for i in _) for _ in self.buf]
        if result not in bases:
            base = ' or '.join(str(_) for _ in bases)
            self.error = f"result {result} != baseline {base}\nRaw timing: {self.buf}"
            return False
        return True


class TestTeeStream(unittest.TestCase):
    def setUp(self):
        self.reenable_gc = gc.isenabled()
        gc.disable()
        gc.collect()
        # Set a short switch interval so that the threading tests behave
        # as expected
        self.switchinterval = sys.getswitchinterval()
        sys.setswitchinterval(tee._poll_interval / 100)

    def tearDown(self):
        sys.setswitchinterval(self.switchinterval)
        if self.reenable_gc:
            gc.enable()
            gc.collect()

    def test_stdout(self):
        a = StringIO()
        b = StringIO()
        with tee.TeeStream(a, b) as t:
            t.STDOUT.write("Hello\n")
        self.assertEqual(a.getvalue(), "Hello\n")
        self.assertEqual(b.getvalue(), "Hello\n")

    def test_err_and_out_are_different(self):
        with tee.TeeStream() as t:
            out = t.STDOUT
            self.assertIs(out, t.STDOUT)
            err = t.STDERR
            self.assertIs(err, t.STDERR)
            self.assertIsNot(out, err)

    @unittest.skipIf(
        not tee._peek_available,
        "Requires the _mergedReader, but _peek_available==False",
    )
    def test_merge_out_and_err(self):
        # Test that the STDERR/STDOUT streams are merged correctly
        # (i.e., STDOUT is line buffered and STDERR is not).  This merge
        # logic is only applicable when using the merged reader (i.e.,
        # _peek_available is True)
        a = StringIO()
        b = StringIO()
        # make sure this doesn't accidentally become a very long wait
        assert tee._poll_interval <= 0.1
        with tee.TeeStream(a, b) as t:
            # This is a slightly nondeterministic (on Windows), so a
            # flush() and short pause should help
            t.STDOUT.write("Hello\nWorld")
            # NOTE: do not flush: we will test flush in the next test
            # t.STDOUT.flush()
            time.sleep(tee._poll_interval * 100)
            t.STDERR.write("interrupting\ncow")
            t.STDERR.flush()
            # For determinism, it is important that the STDERR message
            # appears in the output stream before we start shutting down
            # the TeeStream (which will dump the OUT and ERR in an
            # arbitrary order)
            start_time = time.time()
            while 'cow' not in a.getvalue() and time.time() - start_time < 1:
                time.sleep(tee._poll_interval)
        acceptable_results = {
            "Hello\ninterrupting\ncowWorld",  # expected
            "interrupting\ncowHello\nWorld",  # Windows occasionally puts
            # all error before stdout
        }
        self.assertIn(a.getvalue(), acceptable_results)
        self.assertEqual(b.getvalue(), a.getvalue())

    def test_merge_out_and_err_flush(self):
        # Test that the STDERR/STDOUT streams are merged correctly
        # (i.e., STDOUT is line buffered and STDERR is not).  This merge
        # logic is only applicable when using the merged reader (i.e.,
        # _peek_available is True)
        a = StringIO()
        b = StringIO()
        # make sure this doesn't accidentally become a very long wait
        assert tee._poll_interval <= 0.1
        with tee.TeeStream(a, b) as t:
            # This is a slightly nondeterministic (on Windows), so a
            # flush() and short pause should help
            t.STDOUT.write("Hello\nWorld")
            t.STDOUT.flush()
            time.sleep(tee._poll_interval * 100)
            t.STDERR.write("interrupting\ncow")
            t.STDERR.flush()
            # For determinism, it is important that the STDERR message
            # appears in the output stream before we start shutting down
            # the TeeStream (which will dump the OUT and ERR in an
            # arbitrary order)
            start_time = time.time()
            while 'cow' not in a.getvalue() and time.time() - start_time < 1:
                time.sleep(tee._poll_interval)
        acceptable_results = {
            "Hello\nWorldinterrupting\ncow",  # expected
            "interrupting\ncowHello\nWorld",  # Windows occasionally puts
            # all error before stdout
        }
        self.assertIn(a.getvalue(), acceptable_results)
        self.assertEqual(b.getvalue(), a.getvalue())

    def test_merged_out_and_err_without_peek(self):
        a = StringIO()
        b = StringIO()
        try:
            _tmp, tee._peek_available = tee._peek_available, False
            with tee.TeeStream(a, b) as t:
                # Ensure both threads are running
                t.STDOUT
                t.STDERR
                # ERR should come out before OUT, but this is slightly
                # nondeterministic, so a short pause should help
                t.STDERR.write("Hello\n")
                t.STDERR.flush()
                time.sleep(tee._poll_interval * 2)
                t.STDOUT.write("World\n")
        finally:
            tee._peek_available = _tmp
        self.assertEqual(a.getvalue(), "Hello\nWorld\n")
        self.assertEqual(b.getvalue(), "Hello\nWorld\n")

    def test_binary_tee(self):
        a = BytesIO()
        b = BytesIO()
        with tee.TeeStream(a, b) as t:
            t.open('wb').write(b"Hello\n")
        self.assertEqual(a.getvalue(), b"Hello\n")
        self.assertEqual(b.getvalue(), b"Hello\n")

    def test_decoder_and_buffer_errors(self):
        ref = "Hello, ©"
        bytes_ref = ref.encode()
        log = StringIO()
        with LoggingIntercept(log):
            # Note: we must force the encoding for Windows
            with tee.TeeStream(encoding='utf-8') as t:
                os.write(t.STDOUT.fileno(), bytes_ref[:-1])
        self.assertEqual(
            log.getvalue(),
            "Stream handle closed with a partial line in the output buffer "
            "that was not emitted to the output stream(s):\n"
            "\t'Hello, '\n"
            "Stream handle closed with un-decoded characters in the decoder "
            "buffer that was not emitted to the output stream(s):\n"
            "\tb'\\xc2'\n",
        )

        out = StringIO()
        log = StringIO()
        with LoggingIntercept(log):
            with tee.TeeStream(out) as t:
                out.close()
                t.STDOUT.write("hi\n")
        _id = hex(id(out))
        self.assertRegex(
            log.getvalue(),
            f"Error writing to output stream <_?io.StringIO @ {_id}>:"
            r"\n.*\nOutput stream closed before all output was written to it.\n"
            r"The following was left in the output buffer:\n    'hi\\n'\n$",
        )

        # TeeStream expects stream-like objects
        out = logging.getLogger()
        log = StringIO()
        with LoggingIntercept(log):
            with tee.TeeStream(out) as t:
                t.STDOUT.write("hi\n")
        _id = hex(id(out))
        self.assertRegex(
            log.getvalue(),
            f"Error writing to output stream <logging.RootLogger @ {_id}>:"
            r"\n.*\nIs this a writeable TextIOBase object\?\n"
            r"The following was left in the output buffer:\n    'hi\\n'\n$",
        )

        # Catch partial writes
        class fake_stream:
            def write(self, data):
                return 1

        out = fake_stream()
        log = StringIO()
        with LoggingIntercept(log):
            with tee.TeeStream(out) as t:
                t.STDOUT.write("hi\n")
        _id = hex(id(out))
        self.assertRegex(
            log.getvalue(),
            f"Incomplete write to output stream <.*fake_stream @ {_id}>."
            r"\nThe following was left in the output buffer:\n    'i\\n'\n$",
        )

    def test_capture_output(self):
        out = StringIO()
        with tee.capture_output(out) as OUT:
            print('Hello World')
        self.assertEqual(OUT.getvalue(), 'Hello World\n')

    def test_duplicate_capture_output(self):
        out = StringIO()
        capture = tee.capture_output(out)
        capture.setup()
        try:
            with self.assertRaisesRegex(
                RuntimeError, 'Duplicate call to capture_output.setup'
            ):
                capture.setup()
        finally:
            capture.reset()

    def test_capture_output_logfile_string(self):
        with TempfileManager.new_context() as tempfile:
            logfile = tempfile.create_tempfile()
            self.assertTrue(isinstance(logfile, str))
            with tee.capture_output(logfile):
                print('HELLO WORLD')
            with open(logfile, 'r') as f:
                result = f.read()
            self.assertEqual('HELLO WORLD\n', result)

    def test_capture_output_stack_error(self):
        OUT1 = StringIO()
        OUT2 = StringIO()
        old = (sys.stdout, sys.stderr)
        try:
            a = tee.capture_output(OUT1)
            a.setup()
            b = tee.capture_output(OUT2)
            b.setup()
            with self.assertRaisesRegex(
                RuntimeError, 'Captured output does not match sys.stdout'
            ):
                a.reset()
            b.tee = None
        finally:
            sys.stdout, sys.stderr = old

    def test_capture_output_invalid_ostream(self):
        # Test that capture_output does not suppress errors from the tee
        # module
        _id = hex(id(15))
        with tee.capture_output(capture_fd=True) as OUT:
            with tee.capture_output(15):
                sys.stderr.write("hi\n")
        self.assertEqual(
            OUT.getvalue(),
            f"Error writing to output stream <builtins.int @ {_id}>:\n"
            "    AttributeError: 'int' object has no attribute 'write'\n"
            "Is this a writeable TextIOBase object?\n"
            "The following was left in the output buffer:\n    'hi\\n'\n",
        )

        with tee.capture_output(capture_fd=True) as OUT:
            with tee.capture_output(15, capture_fd=True):
                print("hi")
        self.assertEqual(
            OUT.getvalue(),
            f"Error writing to output stream <builtins.int @ {_id}>:\n"
            "    AttributeError: 'int' object has no attribute 'write'\n"
            "Is this a writeable TextIOBase object?\n"
            "The following was left in the output buffer:\n    'hi\\n'\n",
        )

    def test_deadlock(self):
        class MockStream(object):
            def write(self, data):
                time.sleep(0.2)

            def flush(self):
                pass

        _save = tee._poll_timeout, tee._poll_timeout_deadlock
        tee._poll_timeout = tee._poll_interval * 2**5  # 0.0032
        tee._poll_timeout_deadlock = tee._poll_interval * 2**7  # 0.0128

        try:
            with LoggingIntercept() as LOG, self.assertRaisesRegex(
                RuntimeError, 'deadlock'
            ):
                with tee.TeeStream(MockStream()) as t:
                    err = t.STDERR
                    err.write('*')
            self.assertEqual(
                'Significant delay observed waiting to join reader '
                'threads, possible output stream deadlock\n',
                LOG.getvalue(),
            )
        finally:
            tee._poll_timeout, tee._poll_timeout_deadlock = _save


class BufferTester(object):
    def setUp(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.reenable_gc = gc.isenabled()
        gc.disable()
        gc.collect()
        # Set a short switch interval so that the threading tests behave
        # as expected
        self.switchinterval = sys.getswitchinterval()
        sys.setswitchinterval(tee._poll_interval)
        self.dt = 0.1

    def tearDown(self):
        sys.setswitchinterval(self.switchinterval)
        if self.reenable_gc:
            gc.enable()
            gc.collect()

    def test_buffered_stdout(self):
        # Test 1: short messages to STDOUT are buffered
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}\n")
        with tee.TeeStream(ts, ts) as t, tee.capture_output(t.STDOUT, capture_fd=fd):
            sys.stdout.write(f"{time.time()}\n")
            time.sleep(self.dt)
        ts.write(f"{time.time()}\n")
        baseline = [
            [(0, 0), (1, 0), (1, 0), (1, 1)],
            # TODO: [JDS] The stdout channel appears to sometimes be no
            # longer buffered.  I am not exactly sure why (my guess is
            # because the underlying pipe is not buffered), but as it is
            # generally not a problem to not buffer, we will put off
            # "fixing" it.
            [(0, 0), (0, 0), (0, 0), (1, 1)],
        ]
        if not ts.check(*baseline):
            self.fail(ts.error)

    def test_buffered_stdout_flush(self, retry=True):
        # Test 2: short messages to STDOUT that are flushed are flushed
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}\n")
        with tee.TeeStream(ts, ts) as t, tee.capture_output(t.STDOUT, capture_fd=fd):
            sys.stdout.write(f"{time.time()}\n")
            sys.stdout.flush()
            time.sleep(self.dt)
        ts.write(f"{time.time()}\n")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            # FIXME: We don't know why, but this test will
            # intermittently fail.  For the moment, we will just wait a
            # little and give it a second chance with a longer delay.
            if retry:
                time.sleep(self.dt)
                self.dt *= 2.5
                self.test_buffered_stdout_flush(False)
            elif platform.python_implementation().lower().startswith('pypy'):
                # TODO: For some reason, some part of the flush logic is
                # not reliable under pypy.
                pass
            else:
                self.fail(ts.error)

    def test_buffered_stdout_long_message(self):
        # Test 3: long messages to STDOUT fill the buffer and are flushed
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.TeeStream(ts, ts) as t, tee.capture_output(t.STDOUT, capture_fd=fd):
            sys.stdout.write(f"{time.time()}" + '    ' * 4096 + "\n")
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)

    def test_buffered_stdout_embed_TeeStream(self):
        # Test 4: short messages captured directly to TeeStream are not
        # buffered.
        #
        # TODO: [JDS] I am not exactly sure why this is not buffered (my
        # guess is because the underlying pipe is not buffered), but as
        # it is generally not a problem to not buffer, we will put off
        # "fixing" it.
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.capture_output(tee.TeeStream(ts, ts), capture_fd=fd):
            sys.stdout.write(f"{time.time()}\n")
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)

    def test_buffered_stdout_flush_embed_TeeStream(self):
        # Test 5: short messages captured directly to TeeStream that are
        # flushed are flushed
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.capture_output(tee.TeeStream(ts, ts), capture_fd=fd):
            sys.stdout.write(f"{time.time()}\n")
            sys.stdout.flush()
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)

    def test_buffered_stdout_long_message_embed_TeeStream(self):
        # Test 6: long messages captured directly to TeeStream fill the
        # buffer and are flushed
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.capture_output(tee.TeeStream(ts, ts), capture_fd=fd):
            sys.stdout.write(f"{time.time()}" + '    ' * 4096 + "\n")
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)

    def test_buffered_stderr(self):
        # Test 1: short messages to STDERR are buffered, unless we are
        # capturing the underlying file descriptor, in which case they
        # are buffered.
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.TeeStream(ts, ts) as t, tee.capture_output(t.STDOUT, capture_fd=fd):
            sys.stderr.write(f"{time.time()}\n")
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)

    def test_buffered_stderr_flush(self):
        # Test 2: short messages to STDERR that are flushed are flushed
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.TeeStream(ts, ts) as t, tee.capture_output(t.STDOUT, capture_fd=fd):
            sys.stderr.write(f"{time.time()}\n")
            sys.stderr.flush()
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)

    def test_buffered_stderr_long_message(self):
        # Test 3: long messages to STDERR fill the buffer and are flushed
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.TeeStream(ts, ts) as t, tee.capture_output(t.STDOUT, capture_fd=fd):
            sys.stderr.write(f"{time.time()}" + '  ' * 4096 + "\n")
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)

    def test_buffered_stderr_embed_TeeStream(self):
        # Test 4: short messages captured directly to TeeStream are not
        # buffered, unless we are capturing the underlying file
        # descriptor, in which case they are buffered.
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.capture_output(tee.TeeStream(ts, ts), capture_fd=fd):
            sys.stderr.write(f"{time.time()}\n")
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)

    def test_buffered_stderr_flush_embed_TeeStream(self):
        # Test 5: short messages captured directly to TeeStream that are
        # flushed are flushed
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.capture_output(tee.TeeStream(ts, ts), capture_fd=fd):
            sys.stderr.write(f"{time.time()}\n")
            sys.stderr.flush()
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)

    def test_buffered_stderr_long_message_embed_TeeStream(self):
        # Test 6: long messages captured directly to TeeStream fill the
        # buffer and are flushed
        fd = self.capture_fd
        ts = timestamper()
        ts.write(f"{time.time()}")
        with tee.capture_output(tee.TeeStream(ts, ts), capture_fd=fd):
            sys.stderr.write(f"{time.time()}" + '  ' * 4096 + "\n")
            time.sleep(self.dt)
        ts.write(f"{time.time()}")
        if not ts.check([(0, 0), (0, 0), (0, 0), (1, 1)]):
            self.fail(ts.error)


class TestBuffering_noCapture(BufferTester, unittest.TestCase):
    capture_fd = False


class TestBuffering_capture(BufferTester, unittest.TestCase):
    capture_fd = True


class TestFileDescriptor(unittest.TestCase):
    def setUp(self):
        self.out = sys.stdout
        self.out_fd = os.dup(1)

    def tearDown(self):
        sys.stdout = self.out
        os.dup2(self.out_fd, 1)
        os.close(self.out_fd)

    def _generate_output(self, redirector):
        with redirector:
            sys.stdout.write("to_stdout_1\n")
            sys.stdout.flush()
            with os.fdopen(1, 'w', closefd=False) as F:
                F.write("to_fd1_1\n")
                F.flush()

        sys.stdout.write("to_stdout_2\n")
        sys.stdout.flush()
        with os.fdopen(1, 'w', closefd=False) as F:
            F.write("to_fd1_2\n")
            F.flush()

    def test_redirect_synchronize_stdout(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        sys.stdout = os.fdopen(1, 'w', closefd=False)
        rd = tee.redirect_fd(synchronize=True)
        self._generate_output(rd)

        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), "to_stdout_2\nto_fd1_2\n")

    def test_redirect_no_synchronize_stdout(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        sys.stdout = os.fdopen(1, 'w', closefd=False)
        rd = tee.redirect_fd(synchronize=False)
        self._generate_output(rd)

        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), "to_stdout_1\nto_stdout_2\nto_fd1_2\n")

    # Pytest's default capture method causes failures for the following
    # two tests. This re-implementation of the capfd fixture allows
    # the capture to be disabled for those two test specifically.
    @unittest.pytest.fixture(autouse=True)
    def capfd(self, capfd):
        """
        Reimplementation needed for use in unittest.TestCase subclasses
        """
        self.capfd = capfd

    def test_redirect_synchronize_stdout_not_fd1(self):
        self.capfd.disabled()
        r, w = os.pipe()
        os.dup2(w, 1)
        rd = tee.redirect_fd(synchronize=True)
        self._generate_output(rd)

        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), "to_fd1_2\n")

    def test_redirect_no_synchronize_stdout_not_fd1(self):
        self.capfd.disabled()
        r, w = os.pipe()
        os.dup2(w, 1)
        rd = tee.redirect_fd(synchronize=False)
        self._generate_output(rd)

        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), "to_fd1_2\n")

    def test_redirect_synchronize_stringio(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        try:
            sys.stdout, out = StringIO(), sys.stdout
            rd = tee.redirect_fd(synchronize=True)
            self._generate_output(rd)
        finally:
            sys.stdout, out = out, sys.stdout

        self.assertEqual(out.getvalue(), "to_stdout_2\n")
        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), "to_fd1_2\n")

    def test_redirect_no_synchronize_stringio(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        try:
            sys.stdout, out = StringIO(), sys.stdout
            rd = tee.redirect_fd(synchronize=False)
            self._generate_output(rd)
        finally:
            sys.stdout, out = out, sys.stdout

        self.assertEqual(out.getvalue(), "to_stdout_1\nto_stdout_2\n")
        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), "to_fd1_2\n")

    def test_capture_output_fd(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        sys.stdout = os.fdopen(1, 'w', closefd=False)
        with tee.capture_output(capture_fd=True) as OUT:
            sys.stdout.write("to_stdout_1\n")
            sys.stdout.flush()
            with os.fdopen(1, 'w', closefd=False) as F:
                F.write("to_fd1_1\n")
                F.flush()

        sys.stdout.write("to_stdout_2\n")
        sys.stdout.flush()
        with os.fdopen(1, 'w', closefd=False) as F:
            F.write("to_fd1_2\n")
            F.flush()

        self.assertEqual(OUT.getvalue(), "to_stdout_1\nto_fd1_1\n")
        with os.fdopen(r, 'r') as FILE:
            os.close(1)
            os.close(w)
            self.assertEqual(FILE.read(), "to_stdout_2\nto_fd1_2\n")


if __name__ == '__main__':
    unittest.main()
