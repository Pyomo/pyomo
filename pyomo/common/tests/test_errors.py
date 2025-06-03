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
from pyomo.common.errors import format_exception, ExtendedPyomoException


class LocalException(Exception):
    pass


class CustomLocalException(ExtendedPyomoException):
    message = 'Default message.'


class TestFormatException(unittest.TestCase):
    def test_basic_message(self):
        self.assertEqual(format_exception("Hello world"), "Hello world")

    def test_formatted_message(self):
        self.assertEqual(format_exception("Hello\nworld"), "Hello\nworld")

    def test_long_basic_message(self):
        self.assertEqual(
            format_exception(
                "Hello world, this is a very long message that will "
                "inevitably wrap onto another line."
            ),
            "Hello world, this is a very long message that will\n"
            "    inevitably wrap onto another line.",
        )

    def test_long_basic_message_exception(self):
        self.assertEqual(
            format_exception(
                "Hello world, this is a very long message that will "
                "inevitably wrap onto another line.",
                exception=LocalException(),
            ),
            "Hello world, this is a very\n"
            "    long message that will inevitably wrap onto another line.",
        )

    def test_long_basic_message_builtin_exception(self):
        self.assertEqual(
            format_exception(
                "Hello world, this is a very long message that will "
                "inevitably wrap onto another line.",
                exception=RuntimeError,
            ),
            "Hello world, this is a very long message that will inevitably\n"
            "    wrap onto another line.",
        )

    def test_basic_message_prolog(self):
        self.assertEqual(
            format_exception(
                "This is a very, very, very long message that will "
                "inevitably wrap onto another line.",
                prolog="Hello world:",
            ),
            "Hello world:\n"
            "    This is a very, very, very long message that will inevitably "
            "wrap onto\n"
            "    another line.",
        )

    def test_basic_message_long_prolog(self):
        msg = format_exception(
            "This is a very, very, very long message that will "
            "inevitably wrap onto another line.",
            prolog="Hello, this is a more verbose prolog that will "
            "trigger a line wrap:",
        )
        self.assertEqual(
            msg,
            "Hello, this is a more verbose prolog that will trigger\n"
            "    a line wrap:\n"
            "        This is a very, very, very long message that will inevitably "
            "wrap\n"
            "        onto another line.",
        )

    def test_basic_message_formatted_prolog(self):
        msg = format_exception(
            "This is a very, very, very long message that will "
            "inevitably wrap onto another line.",
            prolog="Hello world:\n    This is a prolog:",
        )
        self.assertEqual(
            msg,
            "Hello world:\n    This is a prolog:\n"
            "        This is a very, very, very long message that will inevitably "
            "wrap\n"
            "        onto another line.",
        )

    def test_basic_message_epilog(self):
        self.assertEqual(
            format_exception(
                "This is a very, very, very long message that will "
                "inevitably wrap onto another line.",
                epilog="Hello world",
            ),
            "This is a very, very, very long message that will\n"
            "        inevitably wrap onto another line.\n"
            "    Hello world",
        )

    def test_basic_message_long_epilog(self):
        self.assertEqual(
            format_exception(
                "This is a very, very, very long message that will "
                "inevitably wrap onto another line.",
                epilog="Hello, this is a very, very, very verbose epilog that will "
                "trigger a line wrap",
            ),
            "This is a very, very, very long message that will\n"
            "        inevitably wrap onto another line.\n"
            "    Hello, this is a very, very, very verbose epilog that will trigger a\n"
            "    line wrap",
        )

    def test_basic_message_formatted_epilog(self):
        msg = format_exception(
            "This is a very, very, very long message that will "
            "inevitably wrap onto another line.",
            epilog="Hello world:\n    This is an epilog:",
        )
        self.assertEqual(
            msg,
            "This is a very, very, very long message that will\n"
            "        inevitably wrap onto another line.\n"
            "Hello world:\n    This is an epilog:",
        )


class TestExtendedPyomoException(unittest.TestCase):
    def test_default_message(self):
        exception = CustomLocalException()
        output = str(exception).replace("\n", " ").strip()
        self.assertIn("Default message.", output)

    def test_custom_message_override(self):
        exception = CustomLocalException("Non-default message.")
        self.assertNotIn("Default message.", str(exception))
        self.assertIn("Non-default message.", str(exception))

    def test_extra_message_epilog(self):
        exception = CustomLocalException(extra_message="Epilog message.")
        self.assertIn("Epilog message.", str(exception))

    def test_prolog_message(self):
        exception = CustomLocalException(prolog="Prolog message.")
        self.assertIn("Prolog message.", str(exception))

    def test_multiple_features(self):
        exception = CustomLocalException(
            "Non-default message.",
            extra_message="Epilog message.",
            prolog="Prolog message.",
        )
        self.assertNotIn("Default message.", str(exception))
        self.assertIn("Non-default message.", str(exception))
        self.assertIn("Epilog message.", str(exception))
        self.assertIn("Prolog message.", str(exception))
