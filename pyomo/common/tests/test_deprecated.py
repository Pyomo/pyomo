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
"""Testing for deprecated function."""
import sys
import types
import weakref

import pyutilib.th as unittest

from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
    deprecated, deprecation_warning, relocated_module_attribute,
)
from pyomo.common.log import LoggingIntercept

from six import StringIO

import logging
logger = logging.getLogger('pyomo.common')


class TestDeprecated(unittest.TestCase):
    """Tests for deprecated function decorator."""

    def test_deprecation_warning(self):
        DEP_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            deprecation_warning(None, version='1.2', remove_in='3.4')

        self.assertIn('DEPRECATED: This has been deprecated',
                      DEP_OUT.getvalue())
        self.assertIn('(deprecated in 1.2, will be removed in 3.4)',
                      DEP_OUT.getvalue().replace('\n',' '))

        DEP_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            deprecation_warning("custom message here", version='1.2', remove_in='3.4')

        self.assertIn('DEPRECATED: custom message here',
                      DEP_OUT.getvalue())
        self.assertIn('(deprecated in 1.2, will be removed in 3.4)',
                      DEP_OUT.getvalue().replace('\n',' '))


    def test_no_version_exception(self):
        with self.assertRaises(DeveloperError):
            @deprecated()
            def foo():
                pass

    def test_no_doc_string(self):
        # Note: No docstring, else nose replaces the function name with
        # the docstring in output.
        #"""Test for deprecated function decorator."""
        @deprecated(version='test')
        def foo(bar='yeah'):
            logger.warning(bar)

        self.assertIn('DEPRECATION WARNING: This function has been deprecated',
                      foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo()
        # Test that the function produces output
        self.assertIn('yeah', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())

        # Test that the function argument gets passed in
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo("custom")
        # Test that the function produces output
        self.assertNotIn('yeah', FCN_OUT.getvalue())
        self.assertIn('custom', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())


    def test_with_doc_string(self):
        @deprecated(version='test')
        def foo(bar='yeah'):
            """Show that I am a good person.

            Because I document my public functions.

            """
            logger.warning(bar)

        self.assertIn('DEPRECATION WARNING: This function has been deprecated',
                      foo.__doc__)
        self.assertIn('I am a good person.', foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo()
        # Test that the function produces output
        self.assertIn('yeah', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())

        # Test that the function argument gets passed in
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo("custom")
        # Test that the function produces output
        self.assertNotIn('yeah', FCN_OUT.getvalue())
        self.assertIn('custom', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())


    def test_with_custom_message(self):
        @deprecated('This is a custom message, too.', version='test')
        def foo(bar='yeah'):
            """Show that I am a good person.

            Because I document my public functions.

            """
            logger.warning(bar)

        self.assertIn('DEPRECATION WARNING: This is a custom message',
                      foo.__doc__)
        self.assertIn('I am a good person.', foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo()
        # Test that the function produces output
        self.assertIn('yeah', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This is a custom message',
                      DEP_OUT.getvalue())

        # Test that the function argument gets passed in
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo("custom")
        # Test that the function produces output
        self.assertNotIn('yeah', FCN_OUT.getvalue())
        self.assertIn('custom', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This is a custom message',
                      DEP_OUT.getvalue())

    def test_with_custom_logger(self):
        @deprecated('This is a custom message', logger='pyomo.common',
                    version='test')
        def foo(bar='yeah'):
            """Show that I am a good person.

            Because I document my public functions.

            """
            logger.warning(bar)

        self.assertIn('DEPRECATION WARNING: This is a custom message',
                      foo.__doc__)
        self.assertIn('I am a good person.', foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo()
        # Test that the function produces output
        self.assertIn('yeah', FCN_OUT.getvalue())
        self.assertIn('DEPRECATED: This is a custom message',
                      FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertNotIn('DEPRECATED:',
                      DEP_OUT.getvalue())

        # Test that the function argument gets passed in
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo("custom")
        # Test that the function produces output
        self.assertNotIn('yeah', FCN_OUT.getvalue())
        self.assertIn('custom', FCN_OUT.getvalue())
        self.assertIn('DEPRECATED: This is a custom message',
                      FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertNotIn('DEPRECATED:', DEP_OUT.getvalue())

    def test_with_class(self):
        @deprecated(version='test')
        class foo(object):
            def __init__(self):
                logger.warning('yeah')

        self.assertIn('DEPRECATION WARNING: This class has been deprecated',
                      foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo()
        # Test that the function produces output
        self.assertIn('yeah', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This class has been deprecated',
                      DEP_OUT.getvalue())


    def test_with_method(self):
        class foo(object):
            def __init__(self):
                pass
            @deprecated(version='test')
            def bar(self):
                logger.warning('yeah')

        self.assertIn('DEPRECATION WARNING: This function has been deprecated',
                      foo.bar.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo().bar()
        # Test that the function produces output
        self.assertIn('yeah', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())

    def test_with_remove_in(self):
        class foo(object):
            def __init__(self):
                pass
            @deprecated(version='1.2', remove_in='3.4')
            def bar(self):
                logger.warning('yeah')

        self.assertIn('DEPRECATION WARNING: This function has been deprecated',
                      foo.bar.__doc__)
        self.assertIn('(deprecated in 1.2, will be removed in 3.4)',
                      foo.bar.__doc__.replace('\n',' '))

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.common'):
                foo().bar()
        # Test that the function produces output
        self.assertIn('yeah', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())
        self.assertIn('(deprecated in 1.2, will be removed in 3.4)',
                      DEP_OUT.getvalue())


class Bar(object):
    data = 21

relocated_module_attribute(
    'myFoo', 'pyomo.common.tests.relocated.Bar', 'test')

class TestRelocated(unittest.TestCase):

    def test_relocated_class(self):
        # Before we test multiple relocated objects, verify that it will
        # handle the import of a new module
        warning = "DEPRECATED: the 'myFoo' class has been moved to " \
                  "'pyomo.common.tests.relocated.Bar'"
        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo.core'):
            from pyomo.common.tests.test_deprecated import myFoo
        self.assertEqual(myFoo.data, 42)
        self.assertIn(warning, OUT.getvalue().replace('\n', ' '))

        from pyomo.common.tests import relocated

        if sys.version_info < (3,5):
            # Make sure that the module is only wrapped once
            self.assertIs(type(relocated._wrapped_module),
                          types.ModuleType)

        self.assertNotIn('Foo', dir(relocated))
        self.assertNotIn('Foo_2', dir(relocated))

        warning = "DEPRECATED: the 'Foo_2' class has been moved to " \
                  "'pyomo.common.tests.relocated.Bar'"

        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo.core'):
            self.assertIs(relocated.Foo_2, relocated.Bar)
            self.assertEqual(relocated.Foo_2.data, 42)
        self.assertIn(warning, OUT.getvalue().replace('\n', ' '))

        self.assertNotIn('Foo', dir(relocated))
        self.assertIn('Foo_2', dir(relocated))
        self.assertIs(relocated.Foo_2, relocated.Bar)

        warning = "DEPRECATED: the 'Foo' class has been moved to " \
                  "'pyomo.common.tests.test_deprecated.Bar'"

        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo.core'):
            from pyomo.common.tests.relocated import Foo
            self.assertEqual(Foo.data, 21)
        self.assertIn(warning, OUT.getvalue().replace('\n', ' '))

        self.assertIn('Foo', dir(relocated))
        self.assertIn('Foo_2', dir(relocated))
        self.assertIs(relocated.Foo, Bar)

        with self.assertRaisesRegex(
                AttributeError,
                "(?:module 'pyomo.common.tests.relocated')|"
                "(?:'module' object) has no attribute 'Baz'"):
            relocated.Baz.data

if __name__ == '__main__':
    unittest.main()
