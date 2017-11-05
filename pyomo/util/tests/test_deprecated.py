"""Testing for deprecated function."""
import pyutilib.th as unittest
from pyomo.util.deprecation import deprecated
from pyomo.util.log import LoggingIntercept

from six import StringIO

import logging
logger = logging.getLogger('pyomo.util')

__author__ = "Qi Chen <https://github.com/qtothec>"

class TestDeprecated(unittest.TestCase):
    """Tests for deprecated function decorator."""

    def test_no_doc_string(self):
        # Note: No docstring, else nose replaces the function name with
        # the docstring in output.
        #"""Test for deprecated function decorator."""
        @deprecated()
        def foo(bar='yeah'):
            logger.warn(bar)

        self.assertIn('DEPRECATION WARNING: This function has been deprecated',
                      foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
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
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
                foo("custom")
        # Test that the function produces output
        self.assertNotIn('yeah', FCN_OUT.getvalue())
        self.assertIn('custom', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())


    def test_with_doc_string(self):
        @deprecated()
        def foo(bar='yeah'):
            """Show that I am a good person.

            Because I document my public functions.

            """
            logger.warn(bar)

        self.assertIn('DEPRECATION WARNING: This function has been deprecated',
                      foo.__doc__)
        self.assertIn('I am a good person.', foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
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
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
                foo("custom")
        # Test that the function produces output
        self.assertNotIn('yeah', FCN_OUT.getvalue())
        self.assertIn('custom', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())


    def test_with_custom_message(self):
        @deprecated('This is a custom message, too.')
        def foo(bar='yeah'):
            """Show that I am a good person.

            Because I document my public functions.

            """
            logger.warn(bar)

        self.assertIn('DEPRECATION WARNING: This is a custom message',
                      foo.__doc__)
        self.assertIn('I am a good person.', foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
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
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
                foo("custom")
        # Test that the function produces output
        self.assertNotIn('yeah', FCN_OUT.getvalue())
        self.assertIn('custom', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This is a custom message',
                      DEP_OUT.getvalue())

    def test_with_custom_logger(self):
        @deprecated('This is a custom message', logger='pyomo.util')
        def foo(bar='yeah'):
            """Show that I am a good person.

            Because I document my public functions.

            """
            logger.warn(bar)

        self.assertIn('DEPRECATION WARNING: This is a custom message',
                      foo.__doc__)
        self.assertIn('I am a good person.', foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
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
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
                foo("custom")
        # Test that the function produces output
        self.assertNotIn('yeah', FCN_OUT.getvalue())
        self.assertIn('custom', FCN_OUT.getvalue())
        self.assertIn('DEPRECATED: This is a custom message',
                      FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertNotIn('DEPRECATED:', DEP_OUT.getvalue())

    def test_with_class(self):
        @deprecated()
        class foo(object):
            def __init__(self):
                logger.warn('yeah')

        self.assertIn('DEPRECATION WARNING: This class has been deprecated',
                      foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
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
            @deprecated()
            def bar(self):
                logger.warn('yeah')

        self.assertIn('DEPRECATION WARNING: This function has been deprecated',
                      foo.bar.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            with LoggingIntercept(FCN_OUT, 'pyomo.util'):
                foo().bar()
        # Test that the function produces output
        self.assertIn('yeah', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())

        

if __name__ == '__main__':
    unittest.main()
