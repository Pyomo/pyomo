# -*- coding: utf-8 -*-
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

import os
import time
import sys

from io import StringIO, BytesIO

from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee


class TestTeeStream(unittest.TestCase):
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
            "Hello\ninterrupting\ncowWorld",  # expected
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
        self.assertRegex(
            log.getvalue(),
            r"^Output stream \(<.*?>\) closed before all output was written "
            r"to it. The following was left in the output buffer:\n\t'hi\\n'\n$",
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

    def test_deadlock(self):
        class MockStream(object):
            def write(self, data):
                time.sleep(0.2)

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
