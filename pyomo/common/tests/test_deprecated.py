"""Testing for deprecated function."""
import pyutilib.th as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import deprecated, deprecation_warning
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
        @deprecated(version='')
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
        @deprecated(version='')
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
        @deprecated('This is a custom message, too.', version='')
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
                    version='')
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
        @deprecated(version='')
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
            @deprecated(version='')
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



if __name__ == '__main__':
    unittest.main()
