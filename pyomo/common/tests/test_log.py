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
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________


import logging
import os
from inspect import currentframe, getframeinfo
from six import StringIO, PY2

import pyutilib.th as unittest

from pyomo.common.log import (
    LoggingIntercept, WrappingFormatter, LegacyPyomoFormatter, LogHandler,
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
                stream = self.stream,
                verbosity=lambda: logger.isEnabledFor(logging.DEBUG)
            )
        self.assertIn('LogHandler class has been deprecated', log.getvalue())
        logger.addHandler(self.handler)

        logger.setLevel(logging.WARNING)
        logger.info("(info)")
        self.assertEqual(self.stream.getvalue(), "")
        logger.warn("(warn)")
        ans = "WARNING: (warn)\n"
        self.assertEqual(self.stream.getvalue(), ans)

        logger.setLevel(logging.DEBUG)
        logger.warn("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_simple_log\n' \
               '    (warn)\n' % (os.path.sep, lineno,)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_default_verbosity(self):
        # Testing positional base, configurable verbosity
        log = StringIO()
        with LoggingIntercept(log):
            self.handler = LogHandler(
                os.path.dirname(__file__),
                stream = self.stream,
            )
        self.assertIn('LogHandler class has been deprecated', log.getvalue())
        logger.addHandler(self.handler)

        logger.setLevel(logging.WARNING)
        logger.warn("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans = 'WARNING: "[base]%stest_log.py", %d, test_default_verbosity\n' \
               '    (warn)\n' % (os.path.sep, lineno,)
        self.assertEqual(self.stream.getvalue(), ans)


class TestWrappingFormatter(unittest.TestCase):
    def setUp(self):
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        logger.addHandler(self.handler)

    def tearDown(self):
        logger.removeHandler(self.handler)

    @unittest.skipIf(PY2, "style added in Python 3.2")
    def test_style_options(self):
        ans = ''

        self.handler.setFormatter(WrappingFormatter(style='%'))
        logger.warn("(warn)")
        ans += "WARNING: (warn)\n"
        self.assertEqual(self.stream.getvalue(), ans)

        self.handler.setFormatter(WrappingFormatter(style='$'))
        logger.warn("(warn)")
        ans += "WARNING: (warn)\n"
        self.assertEqual(self.stream.getvalue(), ans)

        self.handler.setFormatter(WrappingFormatter(style='{'))
        logger.warn("(warn)")
        ans += "WARNING: (warn)\n"
        self.assertEqual(self.stream.getvalue(), ans)

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
        with self.assertRaisesRegex(
                ValueError, "'fmt' is not a valid option"):
            LegacyPyomoFormatter(fmt='%(message)')

        with self.assertRaisesRegex(
                ValueError, "'style' is not a valid option"):
            LegacyPyomoFormatter(style='%')

    def test_simple_log(self):
        # Testing positional base, configurable verbosity
        self.handler.setFormatter(LegacyPyomoFormatter(
            base=os.path.dirname(__file__),
            verbosity=lambda: logger.isEnabledFor(logging.DEBUG)
        ))

        logger.setLevel(logging.WARNING)
        logger.info("(info)")
        self.assertEqual(self.stream.getvalue(), "")
        logger.warn("(warn)")
        ans = "WARNING: (warn)\n"
        self.assertEqual(self.stream.getvalue(), ans)

        logger.setLevel(logging.DEBUG)
        logger.warn("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_simple_log\n' \
               '    (warn)\n' % (os.path.sep, lineno,)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_alternate_base(self):
        self.handler.setFormatter(LegacyPyomoFormatter(
            base = 'log_config',
        ))

        logger.setLevel(logging.WARNING)
        logger.info("(info)")
        self.assertEqual(self.stream.getvalue(), "")
        logger.warn("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans = 'WARNING: "%s", %d, test_alternate_base\n' \
               '    (warn)\n' % (filename, lineno,)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_no_base(self):
        self.handler.setFormatter(LegacyPyomoFormatter())

        logger.setLevel(logging.WARNING)
        logger.info("(info)")
        self.assertEqual(self.stream.getvalue(), "")
        logger.warn("(warn)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans = 'WARNING: "%s", %d, test_no_base\n' \
               '    (warn)\n' % (filename, lineno,)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_no_message(self):
        self.handler.setFormatter(LegacyPyomoFormatter(
            base=os.path.dirname(__file__),
            verbosity=lambda: logger.isEnabledFor(logging.DEBUG)
        ))

        logger.setLevel(logging.WARNING)
        logger.info("")
        self.assertEqual(self.stream.getvalue(), "")

        logger.warn("")
        ans = "WARNING:\n"
        self.assertEqual(self.stream.getvalue(), ans)

        logger.setLevel(logging.DEBUG)
        logger.warn("")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_no_message\n\n' \
               % (os.path.sep, lineno,)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_blank_lines(self):
        self.handler.setFormatter(LegacyPyomoFormatter(
            base=os.path.dirname(__file__),
            verbosity=lambda: logger.isEnabledFor(logging.DEBUG),
        ))

        logger.setLevel(logging.WARNING)
        logger.warn("\n\nthis is a message.\n\n\n")
        ans = "WARNING: this is a message.\n"
        self.assertEqual(self.stream.getvalue(), ans)

        logger.setLevel(logging.DEBUG)
        logger.warn("\n\nthis is a message.\n\n\n")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_blank_lines\n' \
               "    this is a message.\n" % (os.path.sep, lineno)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_numbered_level(self):
        testname ='test_numbered_level'
        self.handler.setFormatter(LegacyPyomoFormatter(
            base=os.path.dirname(__file__),
            verbosity=lambda: logger.isEnabledFor(logging.DEBUG),
        ))

        logger.setLevel(logging.WARNING)
        logger.log(45, "(hi)")
        ans = "Level 45: (hi)\n"
        self.assertEqual(self.stream.getvalue(), ans)

        logger.log(45, "")
        ans += "Level 45:\n"
        self.assertEqual(self.stream.getvalue(), ans)

        logger.setLevel(logging.DEBUG)
        logger.log(45, "(hi)")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'Level 45: "[base]%stest_log.py", %d, %s\n' \
               '    (hi)\n' % (os.path.sep, lineno, testname)
        self.assertEqual(self.stream.getvalue(), ans)

        logger.log(45, "")
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'Level 45: "[base]%stest_log.py", %d, %s\n\n' \
               % (os.path.sep, lineno, testname)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_long_messages(self):
        self.handler.setFormatter(LegacyPyomoFormatter(
            base=os.path.dirname(__file__),
            verbosity=lambda: logger.isEnabledFor(logging.DEBUG)
        ))

        msg = ("This is a long message\n"
               "\n"
               "With some kind of internal formatting\n"
               "    - including a bulleted list\n"
               "    - list 2  ")
        logger.setLevel(logging.WARNING)
        logger.warn(msg)
        ans = ( "WARNING: This is a long message\n"
                "\n"
                "    With some kind of internal formatting\n"
                "        - including a bulleted list\n"
                "        - list 2\n" )
        self.assertEqual(self.stream.getvalue(), ans)

        logger.setLevel(logging.DEBUG)
        logger.info(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += ( 'INFO: "[base]%stest_log.py", %d, test_long_messages\n'
                 "    This is a long message\n"
                 "\n"
                 "    With some kind of internal formatting\n"
                 "        - including a bulleted list\n"
                 "        - list 2\n" % (os.path.sep, lineno,))
        self.assertEqual(self.stream.getvalue(), ans)

        # test trailing newline
        msg += "\n"
        logger.setLevel(logging.WARNING)
        logger.warn(msg)
        ans += ( "WARNING: This is a long message\n"
                 "\n"
                 "    With some kind of internal formatting\n"
                 "        - including a bulleted list\n"
                 "        - list 2\n" )
        self.assertEqual(self.stream.getvalue(), ans)

        logger.setLevel(logging.DEBUG)
        logger.info(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += ( 'INFO: "[base]%stest_log.py", %d, test_long_messages\n'
                 "    This is a long message\n"
                 "\n"
                 "    With some kind of internal formatting\n"
                 "        - including a bulleted list\n"
                 "        - list 2\n" % (os.path.sep, lineno,))
        self.assertEqual(self.stream.getvalue(), ans)

        # test initial and final blank lines
        msg = "\n" + msg + "\n\n"
        logger.setLevel(logging.WARNING)
        logger.warn(msg)
        ans += ( "WARNING: This is a long message\n"
                 "\n"
                "    With some kind of internal formatting\n"
                "        - including a bulleted list\n"
                "        - list 2\n" )
        self.assertEqual(self.stream.getvalue(), ans)

        logger.setLevel(logging.DEBUG)
        logger.info(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += ( 'INFO: "[base]%stest_log.py", %d, test_long_messages\n'
                 "    This is a long message\n"
                 "\n"
                 "    With some kind of internal formatting\n"
                 "        - including a bulleted list\n"
                 "        - list 2\n" % (os.path.sep, lineno,))
        self.assertEqual(self.stream.getvalue(), ans)

    def test_verbatim(self):
        self.handler.setFormatter(LegacyPyomoFormatter(
            base=os.path.dirname(__file__),
            verbosity=lambda: logger.isEnabledFor(logging.DEBUG)
        ))

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
        )
        logger.setLevel(logging.WARNING)
        logger.warn(msg)
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
        )
        self.assertEqual(self.stream.getvalue(), ans)
