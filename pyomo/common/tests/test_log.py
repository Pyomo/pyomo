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
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________


import logging
import os
import sys
from inspect import currentframe, getframeinfo
from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.log import (
    LegacyPyomoFormatter,
    LoggingIntercept,
    LogHandler,
    LogStream,
    Preformatted,
    StdoutHandler,
    WrappingFormatter,
    is_debug_set,
    pyomo_formatter,
)

logger = logging.getLogger('pyomo.common.log.testing')
filename = getframeinfo(currentframe()).filename


class TestLegacyLogHandler(unittest.TestCase):
    def setUp(self):
        self.stream = StringIO()

    def tearDown(self):
        logger.removeHandler(self.handler)

    def test_simple_log(self):
        # Testing positional base, configurable verbosity
        log = StringIO()
        with LoggingIntercept(log):
            self.handler = LogHandler(
                os.path.dirname(__file__),
                stream=self.stream,
                verbosity=lambda: is_debug_set(logger),
            )
        self.assertIn('LogHandler class has been deprecated', log.getvalue())
        logger.addHandler(self.handler)

        logger.setLevel(logging.WARNING)
        logger.info("(info)")
        self.assertEqual("", self.stream.getvalue())
        logger.warning("(warn)")
        ans = "WARNING: (warn)\n"
        self.assertEqual(ans, self.stream.getvalue())

        logger.setLevel(logging.DEBUG)
        logger.warning("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += (
            'WARNING: "[base]%stest_log.py", %d, test_simple_log\n'
            '    (warn)\n' % (os.path.sep, lineno)
        )
        self.assertEqual(ans, self.stream.getvalue())

    def test_default_verbosity(self):
        # Testing positional base, configurable verbosity
        log = StringIO()
        with LoggingIntercept(log):
            self.handler = LogHandler(os.path.dirname(__file__), stream=self.stream)
        self.assertIn('LogHandler class has been deprecated', log.getvalue())
        logger.addHandler(self.handler)

        logger.setLevel(logging.WARNING)
        logger.warning("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans = (
            'WARNING: "[base]%stest_log.py", %d, test_default_verbosity\n'
            '    (warn)\n' % (os.path.sep, lineno)
        )
        self.assertEqual(ans, self.stream.getvalue())


class TestWrappingFormatter(unittest.TestCase):
    def setUp(self):
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        logger.addHandler(self.handler)

    def tearDown(self):
        logger.removeHandler(self.handler)

    def test_style_options(self):
        ans = ''

        self.handler.setFormatter(WrappingFormatter(style='%'))
        logger.warning("(warn)")
        ans += "WARNING: (warn)\n"
        self.assertEqual(ans, self.stream.getvalue())

        self.handler.setFormatter(WrappingFormatter(style='$'))
        logger.warning("(warn)")
        ans += "WARNING: (warn)\n"
        self.assertEqual(ans, self.stream.getvalue())

        self.handler.setFormatter(WrappingFormatter(style='{'))
        logger.warning("(warn)")
        ans += "WARNING: (warn)\n"
        self.assertEqual(ans, self.stream.getvalue())

        with self.assertRaisesRegex(ValueError, 'unrecognized style flag "s"'):
            WrappingFormatter(style='s')


class TestLegacyPyomoFormatter(unittest.TestCase):
    def setUp(self):
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        logger.addHandler(self.handler)

    def tearDown(self):
        logger.removeHandler(self.handler)

    def test_unallowed_options(self):
        with self.assertRaisesRegex(ValueError, "'fmt' is not a valid option"):
            LegacyPyomoFormatter(fmt='%(message)')

        with self.assertRaisesRegex(ValueError, "'style' is not a valid option"):
            LegacyPyomoFormatter(style='%')

    def test_simple_log(self):
        # Testing positional base, configurable verbosity
        self.handler.setFormatter(
            LegacyPyomoFormatter(
                base=os.path.dirname(__file__), verbosity=lambda: is_debug_set(logger)
            )
        )

        logger.setLevel(logging.WARNING)
        logger.info("(info)")
        self.assertEqual("", self.stream.getvalue())
        logger.warning("(warn)")
        ans = "WARNING: (warn)\n"
        self.assertEqual(ans, self.stream.getvalue())

        logger.setLevel(logging.DEBUG)
        logger.warning("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += (
            'WARNING: "[base]%stest_log.py", %d, test_simple_log\n'
            '    (warn)\n' % (os.path.sep, lineno)
        )
        self.assertEqual(ans, self.stream.getvalue())

    def test_alternate_base(self):
        self.handler.setFormatter(LegacyPyomoFormatter(base='log_config'))

        logger.setLevel(logging.WARNING)
        logger.info("(info)")
        self.assertEqual("", self.stream.getvalue())
        logger.warning("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans = 'WARNING: "%s", %d, test_alternate_base\n    (warn)\n' % (
            filename,
            lineno,
        )
        self.assertEqual(ans, self.stream.getvalue())

    def test_no_base(self):
        self.handler.setFormatter(LegacyPyomoFormatter())

        logger.setLevel(logging.WARNING)
        logger.info("(info)")
        self.assertEqual("", self.stream.getvalue())
        logger.warning("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans = 'WARNING: "%s", %d, test_no_base\n    (warn)\n' % (filename, lineno)
        self.assertEqual(ans, self.stream.getvalue())

    def test_no_message(self):
        self.handler.setFormatter(
            LegacyPyomoFormatter(
                base=os.path.dirname(__file__), verbosity=lambda: is_debug_set(logger)
            )
        )

        logger.setLevel(logging.WARNING)
        logger.info("")
        self.assertEqual("", self.stream.getvalue())

        logger.warning("")
        ans = "WARNING:\n"
        self.assertEqual(ans, self.stream.getvalue())

        logger.setLevel(logging.DEBUG)
        logger.warning("")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_no_message\n\n' % (
            os.path.sep,
            lineno,
        )
        self.assertEqual(ans, self.stream.getvalue())

    def test_blank_lines(self):
        self.handler.setFormatter(
            LegacyPyomoFormatter(
                base=os.path.dirname(__file__), verbosity=lambda: is_debug_set(logger)
            )
        )

        logger.setLevel(logging.WARNING)
        logger.warning("\n\nthis is a message.\n\n\n")
        ans = "WARNING: this is a message.\n"
        self.assertEqual(ans, self.stream.getvalue())

        logger.setLevel(logging.DEBUG)
        logger.warning("\n\nthis is a message.\n\n\n")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += (
            'WARNING: "[base]%stest_log.py", %d, test_blank_lines\n'
            "    this is a message.\n" % (os.path.sep, lineno)
        )
        self.assertEqual(ans, self.stream.getvalue())

    def test_numbered_level(self):
        testname = 'test_numbered_level'
        self.handler.setFormatter(
            LegacyPyomoFormatter(
                base=os.path.dirname(__file__), verbosity=lambda: is_debug_set(logger)
            )
        )

        logger.setLevel(logging.WARNING)
        logger.log(45, "(hi)")
        ans = "Level 45: (hi)\n"
        self.assertEqual(ans, self.stream.getvalue())

        logger.log(45, "")
        ans += "Level 45:\n"
        self.assertEqual(ans, self.stream.getvalue())

        logger.setLevel(logging.DEBUG)
        logger.log(45, "(hi)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'Level 45: "[base]%stest_log.py", %d, %s\n    (hi)\n' % (
            os.path.sep,
            lineno,
            testname,
        )
        self.assertEqual(ans, self.stream.getvalue())

        logger.log(45, "")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'Level 45: "[base]%stest_log.py", %d, %s\n\n' % (
            os.path.sep,
            lineno,
            testname,
        )
        self.assertEqual(ans, self.stream.getvalue())

    def test_preformatted(self):
        self.handler.setFormatter(
            LegacyPyomoFormatter(
                base=os.path.dirname(__file__), verbosity=lambda: is_debug_set(logger)
            )
        )

        msg = """This is a long multi-line message that in normal circumstances \
would be line-wrapped
        with additional information
        that normally would be combined."""

        logger.setLevel(logging.WARNING)
        logger.info(msg)
        self.assertEqual("", self.stream.getvalue())

        logger.warning(Preformatted(msg))
        ans = msg + "\n"
        self.assertEqual(ans, self.stream.getvalue())

        logger.warning(msg)
        ans += (
            "WARNING: This is a long multi-line message that in normal "
            "circumstances would\n"
            "be line-wrapped with additional information that normally would be combined.\n"
        )
        self.assertEqual(ans, self.stream.getvalue())

        logger.setLevel(logging.DEBUG)

        logger.warning(Preformatted(msg))
        ans += msg + "\n"
        self.assertEqual(ans, self.stream.getvalue())

        logger.warning(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_preformatted\n' % (
            os.path.sep,
            lineno,
        )
        ans += (
            "    This is a long multi-line message that in normal "
            "circumstances would be\n"
            "    line-wrapped with additional information that normally would be combined.\n"
        )
        self.assertEqual(ans, self.stream.getvalue())

    def test_long_messages(self):
        self.handler.setFormatter(
            LegacyPyomoFormatter(
                base=os.path.dirname(__file__), verbosity=lambda: is_debug_set(logger)
            )
        )

        msg = (
            "This is a long message\n"
            "\n"
            "With some kind of internal formatting\n"
            "    - including a bulleted list\n"
            "    - list 2  "
        )
        logger.setLevel(logging.WARNING)
        logger.warning(msg)
        ans = (
            "WARNING: This is a long message\n"
            "\n"
            "    With some kind of internal formatting\n"
            "        - including a bulleted list\n"
            "        - list 2\n"
        )
        self.assertEqual(ans, self.stream.getvalue())

        logger.setLevel(logging.DEBUG)
        logger.info(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += (
            'INFO: "[base]%stest_log.py", %d, test_long_messages\n'
            "    This is a long message\n"
            "\n"
            "    With some kind of internal formatting\n"
            "        - including a bulleted list\n"
            "        - list 2\n" % (os.path.sep, lineno)
        )
        self.assertEqual(ans, self.stream.getvalue())

        # test trailing newline
        msg += "\n"
        logger.setLevel(logging.WARNING)
        logger.warning(msg)
        ans += (
            "WARNING: This is a long message\n"
            "\n"
            "    With some kind of internal formatting\n"
            "        - including a bulleted list\n"
            "        - list 2\n"
        )
        self.assertEqual(ans, self.stream.getvalue())

        logger.setLevel(logging.DEBUG)
        logger.info(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += (
            'INFO: "[base]%stest_log.py", %d, test_long_messages\n'
            "    This is a long message\n"
            "\n"
            "    With some kind of internal formatting\n"
            "        - including a bulleted list\n"
            "        - list 2\n" % (os.path.sep, lineno)
        )
        self.assertEqual(ans, self.stream.getvalue())

        # test initial and final blank lines
        msg = "\n" + msg + "\n\n"
        logger.setLevel(logging.WARNING)
        logger.warning(msg)
        ans += (
            "WARNING: This is a long message\n"
            "\n"
            "    With some kind of internal formatting\n"
            "        - including a bulleted list\n"
            "        - list 2\n"
        )
        self.assertEqual(ans, self.stream.getvalue())

        logger.setLevel(logging.DEBUG)
        logger.info(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += (
            'INFO: "[base]%stest_log.py", %d, test_long_messages\n'
            "    This is a long message\n"
            "\n"
            "    With some kind of internal formatting\n"
            "        - including a bulleted list\n"
            "        - list 2\n" % (os.path.sep, lineno)
        )
        self.assertEqual(ans, self.stream.getvalue())

    def test_verbatim(self):
        self.handler.setFormatter(
            LegacyPyomoFormatter(
                base=os.path.dirname(__file__), verbosity=lambda: is_debug_set(logger)
            )
        )

        msg = (
            "This is a long message\n"
            "\n"
            "   ```\n"
            "With some \n"
            "internal\n"
            "verbatim \n"
            "  - including a\n"
            "    long list\n"
            "  - and a short list \n"
            "  ```\n"
            "\n"
            "And some \n"
            "internal\n"
            "non-verbatim \n"
            "  - including a\n"
            "    long list\n"
            "  - and a short list \n"
            "\n"
            "And a section\n"
            "~~~~~~~~~~~~~\n"
            "\n"
            "  | and\n"
            "  | a line\n"
            "  | block\n"
            "\n"
            "And a\n"
            "quoted literal::\n"
            "\n"
            ">> he said\n"
            ">\n"
            "> and they replied\n"
            "\n"
            "this is\n"
            "outside the quote\n"
            "\n"
            "indented literal::\n"
            "\n"
            "    Here is\n"
            "       an indented\n"
            "\n"
            "    literal\n"
            "    with a blank line\n"
            "\n"
            "Finally, an invalid::\n"
            "\n"
            "quote\n"
            "block\n"
        )
        logger.setLevel(logging.WARNING)
        logger.warning(msg)
        ans = (
            "WARNING: This is a long message\n"
            "\n"
            "    With some \n"
            "    internal\n"
            "    verbatim \n"
            "      - including a\n"
            "        long list\n"
            "      - and a short list \n"
            "\n"
            "    And some internal non-verbatim\n"
            "      - including a long list\n"
            "      - and a short list\n"
            "\n"
            "    And a section\n"
            "    ~~~~~~~~~~~~~\n"
            "\n"
            "      | and\n"
            "      | a line\n"
            "      | block\n"
            "\n"
            "    And a quoted literal::\n"
            "\n"
            "    >> he said\n"
            "    >\n"
            "    > and they replied\n"
            "\n"
            "    this is outside the quote\n"
            "\n"
            "    indented literal::\n"
            "\n"
            "        Here is\n"
            "           an indented\n"
            "\n"
            "        literal\n"
            "        with a blank line\n"
            "\n"
            "    Finally, an invalid::\n"
            "\n"
            "    quote block\n"
        )
        self.assertEqual(ans, self.stream.getvalue())


class TestLogStream(unittest.TestCase):
    def test_log_stream(self):
        ls = LogStream(logging.INFO, logging.getLogger('pyomo'))
        LI = LoggingIntercept(level=logging.INFO, formatter=pyomo_formatter)
        with LI as OUT:
            ls.write("hello, world\n")
            self.assertEqual(OUT.getvalue(), "INFO: hello, world\n")

        with LI as OUT:
            ls.write("line 1\nline 2\n")
            self.assertEqual(OUT.getvalue(), "INFO: line 1\nINFO: line 2\n")

        with LI as OUT:
            # empty writes do not generate log records
            ls.write("")
            ls.flush()
            self.assertEqual("", OUT.getvalue())

        with LI as OUT:
            ls.write("line 1\nline 2")
            self.assertEqual(OUT.getvalue(), "INFO: line 1\n")

        with LI as OUT:
            ls.flush()
            self.assertEqual(OUT.getvalue(), "INFO: line 2\n")
            # Second flush should do nothing
            ls.flush()
            self.assertEqual(OUT.getvalue(), "INFO: line 2\n")

        with LI as OUT:
            with LogStream(logging.INFO, logging.getLogger('pyomo')) as ls:
                ls.write('line 1\nline 2')
                self.assertEqual(OUT.getvalue(), "INFO: line 1\n")
            # Exiting the context manager flushes the LogStream
            self.assertEqual(OUT.getvalue(), "INFO: line 1\nINFO: line 2\n")

    def test_loggerAdapter(self):
        class Adapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                return '[%s] %s' % (self.extra['foo'], msg), kwargs

        adapter = Adapter(logging.getLogger('pyomo'), {"foo": 42})
        ls = LogStream(logging.INFO, adapter)
        LI = LoggingIntercept(level=logging.INFO, formatter=pyomo_formatter)
        with LI as OUT:
            ls.write("hello, world\n")
            self.assertEqual(OUT.getvalue(), "INFO: [42] hello, world\n")


class TestPreformatted(unittest.TestCase):
    def test_preformatted_api(self):
        ref = 'a message'
        msg = Preformatted(ref)
        self.assertIs(msg.msg, ref)
        self.assertEqual(str(msg), ref)
        self.assertEqual(repr(msg), "Preformatted('a message')")

        ref = 2
        msg = Preformatted(ref)
        self.assertIs(msg.msg, ref)
        self.assertEqual(str(msg), '2')
        self.assertEqual(repr(msg), "Preformatted(2)")


class TestLoggingIntercept(unittest.TestCase):
    def test_init(self):
        li = LoggingIntercept()
        self.assertEqual(li.module, 'root')

        li = LoggingIntercept(module='pyomo.core')
        self.assertEqual(li.module, 'pyomo.core')

        li = LoggingIntercept(logger=logger)
        self.assertEqual(li.module, 'pyomo.common.log.testing')

        with self.assertRaisesRegex(
            ValueError, "LoggingIntercept: only one of 'module' and 'logger' is allowed"
        ):
            li = LoggingIntercept(module='pyomo', logger=logger)

    def test_propagate(self):
        self.assertEqual(logger.propagate, True)
        with LoggingIntercept(logger=logger):
            self.assertEqual(logger.propagate, False)
        self.assertEqual(logger.propagate, True)

    def test_propagate(self):
        self.assertEqual(logger.level, 30)
        try:
            with LoggingIntercept(logger=logger, level=None):
                self.assertEqual(logger.level, 30)
            self.assertEqual(logger.level, 30)
            with LoggingIntercept(logger=logger, level=40):
                self.assertEqual(logger.level, 40)
            self.assertEqual(logger.level, 30)

            logger.setLevel(40)
            with LoggingIntercept(logger=logger, level=None):
                self.assertEqual(logger.level, 40)
            self.assertEqual(logger.level, 40)
            with LoggingIntercept(logger=logger, level=30):
                self.assertEqual(logger.level, 30)
            self.assertEqual(logger.level, 40)
        finally:
            logger.setLevel(30)


class TestStdoutHandler(unittest.TestCase):
    def setUp(self):
        self.orig = sys.stdout

    def tearDown(self):
        sys.stdout = self.orig

    def test_emit(self):
        handler = StdoutHandler()
        self.assertIsNone(handler.stream)

        sys.stdout = StringIO()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test msg",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        handler.flush()

        self.assertEqual(sys.stdout.getvalue(), "Test msg\n")
        self.assertIsNone(handler.stream)

    def test_handler(self):
        logger = logging.getLogger(__name__)
        propagate, level = logger.propagate, logger.level
        handler = StdoutHandler()
        try:
            logger.addHandler(handler)
            logger.propagate = False
            sys.stdout = StringIO()

            logger.setLevel(logging.WARNING)
            logger.info("Test1")
            self.assertEqual("", sys.stdout.getvalue())

            logger.setLevel(logging.INFO)
            logger.info("Test2")
            self.assertEqual(sys.stdout.getvalue(), "Test2\n")
        finally:
            logger.removeHandler(handler)
            logger.propagate = propagate
            logger.setLevel(level)
