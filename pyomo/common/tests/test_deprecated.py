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

import pyomo.common.unittest as unittest

from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
    deprecated, deprecation_warning, relocated_module_attribute, RenamedClass,
    _import_object
)
from pyomo.common.log import LoggingIntercept

from io import StringIO

import logging
logger = logging.getLogger('local')


class TestDeprecated(unittest.TestCase):
    """Tests for deprecated function decorator."""

    def test_deprecation_warning(self):
        DEP_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            deprecation_warning(None, version='1.2', remove_in='3.4')

        self.assertIn('DEPRECATED: This has been deprecated',
                      DEP_OUT.getvalue())
        self.assertIn('(deprecated in 1.2, will be removed in (or after) 3.4)',
                      DEP_OUT.getvalue().replace('\n',' '))

        DEP_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            deprecation_warning("custom message here", version='1.2', remove_in='3.4')

        self.assertIn('DEPRECATED: custom message here',
                      DEP_OUT.getvalue())
        self.assertIn('(deprecated in 1.2, will be removed in (or after) 3.4)',
                      DEP_OUT.getvalue().replace('\n',' '))


    def test_no_version_exception(self):
        with self.assertRaisesRegex(
                DeveloperError, "@deprecated missing initial version"):
            @deprecated()
            def foo():
                pass

        with self.assertRaisesRegex(
                DeveloperError, "@deprecated missing initial version"):
            @deprecated()
            class foo(object):
                pass

        # But no exception if the class can infer a version from the
        # __init__ (or __new__ or __new_member__)
        @deprecated()
        class foo(object):
            @deprecated(version="1.2")
            def __init__(self):
                pass
        self.assertIn('.. deprecated:: 1.2', foo.__doc__)

    def test_no_doc_string(self):
        # Note: No docstring, else nose replaces the function name with
        # the docstring in output.
        #"""Test for deprecated function decorator."""
        @deprecated(version='test')
        def foo(bar='yeah'):
            logger.warning(bar)

        self.assertIn(
            '.. deprecated:: test\n   This function has been deprecated',
            foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
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
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
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

        self.assertIn(
            '.. deprecated:: test\n   This function has been deprecated',
            foo.__doc__)
        self.assertIn('I am a good person.', foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
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
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
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

        self.assertIn(
            '.. deprecated:: test\n   This is a custom message',
            foo.__doc__)
        self.assertIn('I am a good person.', foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
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
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
                foo("custom")
        # Test that the function produces output
        self.assertNotIn('yeah', FCN_OUT.getvalue())
        self.assertIn('custom', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This is a custom message',
                      DEP_OUT.getvalue())


    def test_with_custom_logger(self):
        @deprecated('This is a custom message', logger='local',
                    version='test')
        def foo(bar='yeah'):
            """Show that I am a good person.

            Because I document my public functions.

            """
            logger.warning(bar)

        self.assertIn(
            '.. deprecated:: test\n   This is a custom message',
            foo.__doc__)
        self.assertIn('I am a good person.', foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
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
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
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

        self.assertIs(type(foo), type)
        self.assertIn(
            '.. deprecated:: test\n   This class has been deprecated',
            foo.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
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

        self.assertIn(
            '.. deprecated:: test\n   This function has been deprecated',
            foo.bar.__doc__)

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
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

        self.assertIn(
            '.. deprecated:: 1.2\n   This function has been deprecated',
            foo.bar.__doc__)
        self.assertIn('(will be removed in (or after) 3.4)',
                      foo.bar.__doc__.replace('\n',' '))

        # Test the default argument
        DEP_OUT = StringIO()
        FCN_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo'):
            with LoggingIntercept(FCN_OUT, 'local'):
                foo().bar()
        # Test that the function produces output
        self.assertIn('yeah', FCN_OUT.getvalue())
        self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
        # Test that the deprecation warning was logged
        self.assertIn('DEPRECATED: This function has been deprecated',
                      DEP_OUT.getvalue())
        self.assertIn('(deprecated in 1.2, will be removed in (or after) 3.4)',
                      DEP_OUT.getvalue().replace('\n', ' '))


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
        with LoggingIntercept(OUT, 'pyomo'):
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
        with LoggingIntercept(OUT, 'pyomo'):
            self.assertIs(relocated.Foo_2, relocated.Bar)
            self.assertEqual(relocated.Foo_2.data, 42)
        self.assertIn(warning, OUT.getvalue().replace('\n', ' '))

        self.assertNotIn('Foo', dir(relocated))
        self.assertIn('Foo_2', dir(relocated))
        self.assertIs(relocated.Foo_2, relocated.Bar)

        warning = "DEPRECATED: the 'Foo' class has been moved to " \
                  "'pyomo.common.tests.test_deprecated.Bar'"

        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo'):
            from pyomo.common.tests.relocated import Foo
            self.assertEqual(Foo.data, 21)
        self.assertIn(warning, OUT.getvalue().replace('\n', ' '))

        self.assertIn('Foo', dir(relocated))
        self.assertIn('Foo_2', dir(relocated))
        self.assertIs(relocated.Foo, Bar)

        # Note that relocated defines a __getattr__, which changes how
        # attribute processing is handled in python 3.7+
        with self.assertRaisesRegex(
                AttributeError,
                "(?:module 'pyomo.common.tests.relocated')|"
                "(?:'module' object) has no attribute 'Baz'"):
            relocated.Baz.data
        if sys.version_info[:2] >= (3, 7):
            self.assertEqual(relocated.Foo_3, '_3')

        with self.assertRaisesRegex(
                AttributeError,
                "(?:module 'pyomo.common.tests.test_deprecated')|"
                "(?:'module' object) has no attribute 'Baz'"):
            sys.modules[__name__].Baz.data


    def test_relocated_message(self):
        with LoggingIntercept() as LOG:
            self.assertIs(_import_object(
                'oldName', 'pyomo.common.tests.test_deprecated.logger',
                'TBD', None), logger)
        self.assertRegex(
            LOG.getvalue().replace('\n', ' '),
            "DEPRECATED: the 'oldName' attribute has been moved to "
            "'pyomo.common.tests.test_deprecated.logger'")

        with LoggingIntercept() as LOG:
            self.assertIs(_import_object(
                'oldName', 'pyomo.common.tests.test_deprecated._import_object',
                'TBD', None), _import_object)
        self.assertRegex(
            LOG.getvalue().replace('\n', ' '),
            "DEPRECATED: the 'oldName' function has been moved to "
            "'pyomo.common.tests.test_deprecated._import_object'")

        with LoggingIntercept() as LOG:
            self.assertIs(_import_object(
                'oldName', 'pyomo.common.tests.test_deprecated.TestRelocated',
                'TBD', None), TestRelocated)
        self.assertRegex(
            LOG.getvalue().replace('\n', ' '),
            "DEPRECATED: the 'oldName' class has been moved to "
            "'pyomo.common.tests.test_deprecated.TestRelocated'")


class TestRenamedClass(unittest.TestCase):
    def test_renamed(self):
        class NewClass(object):
            attr = 'NewClass'

        class NewClassSubclass(NewClass):
            pass

        # The deprecated class does not generate a warning
        out = StringIO()
        with LoggingIntercept(out):
            class DeprecatedClass(metaclass=RenamedClass):
                __renamed__new_class__ = NewClass
                __renamed__version__ = 'X.y'
        self.assertEqual(out.getvalue(), "")

        # Inheriting from the deprecated class generates the warning
        out = StringIO()
        with LoggingIntercept(out):
            class DeprecatedClassSubclass(DeprecatedClass):
                attr = 'DeprecatedClassSubclass'
        self.assertRegex(
            out.getvalue().replace("\n", " ").strip(),
            r"^DEPRECATED: Declaring class 'DeprecatedClassSubclass' "
            r"derived from 'DeprecatedClass'.  "
            r"The class 'DeprecatedClass' has been renamed to 'NewClass'.  "
            r"\(deprecated in X.y\) \(called from [^\)]*\)$",
        )

        # Inheriting from a class derived from the deprecated class does
        # not generate a warning
        out = StringIO()
        with LoggingIntercept(out):
            class DeprecatedClassSubSubclass(DeprecatedClassSubclass):
                attr = 'DeprecatedClassSubSubclass'
        self.assertEqual(out.getvalue(), "")

        #
        # Test class creation
        #

        out = StringIO()
        with LoggingIntercept(out):
            newclass = NewClass()
            newclasssubclass = NewClassSubclass()
        self.assertEqual(out.getvalue(), "")

        out = StringIO()
        with LoggingIntercept(out):
            deprecatedclass = DeprecatedClass()
        self.assertRegex(
            out.getvalue().replace("\n", " ").strip(),
            r"^DEPRECATED: Instantiating class 'DeprecatedClass'.  "
            r"The class 'DeprecatedClass' has been renamed to 'NewClass'.  "
            r"\(deprecated in X.y\) \(called from [^\)]*\)$",
        )

        # Instantiating a class derived from the deprecaed class does
        # not generate a warning (the warning is generated when the
        # class is declared)
        out = StringIO()
        with LoggingIntercept(out):
            deprecatedsubclass = DeprecatedClassSubclass()
            deprecatedsubsubclass = DeprecatedClassSubSubclass()
        self.assertEqual(out.getvalue(), "")

        #
        # Test isinstance
        #
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIsInstance(deprecatedsubclass, NewClass)
            self.assertIsInstance(deprecatedsubsubclass, NewClass)
        self.assertEqual(out.getvalue(), "")

        for obj in (newclass, newclasssubclass, deprecatedclass,
                    deprecatedsubclass, deprecatedsubsubclass):
            out = StringIO()
            with LoggingIntercept(out):
                self.assertIsInstance(obj, DeprecatedClass)
            self.assertRegex(
                out.getvalue().replace("\n", " ").strip(),
                r"^DEPRECATED: Checking type relative to 'DeprecatedClass'.  "
                r"The class 'DeprecatedClass' has been renamed to 'NewClass'."
                r"  \(deprecated in X.y\) \(called from [^\)]*\)$",
            )

        #
        # Test issubclass
        #

        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(issubclass(DeprecatedClass, NewClass))
            self.assertTrue(issubclass(DeprecatedClassSubclass, NewClass))
            self.assertTrue(issubclass(DeprecatedClassSubSubclass, NewClass))
        self.assertEqual(out.getvalue(), "")

        for cls in (NewClass, NewClassSubclass, DeprecatedClass,
                    DeprecatedClassSubclass, DeprecatedClassSubSubclass):
            out = StringIO()
            with LoggingIntercept(out):
                self.assertTrue(issubclass(cls, DeprecatedClass))
            self.assertRegex(
                out.getvalue().replace("\n", " ").strip(),
                r"^DEPRECATED: Checking type relative to 'DeprecatedClass'.  "
                r"The class 'DeprecatedClass' has been renamed to 'NewClass'."
                r"  \(deprecated in X.y\) \(called from [^\)]*\)$",
            )

        #
        # Test class attributes
        #
        self.assertEqual(newclass.attr, 'NewClass')
        self.assertEqual(newclasssubclass.attr, 'NewClass')
        self.assertEqual(deprecatedclass.attr, 'NewClass')
        self.assertEqual(deprecatedsubclass.attr,
                         'DeprecatedClassSubclass')
        self.assertEqual(deprecatedsubsubclass.attr,
                         'DeprecatedClassSubSubclass')
        self.assertEqual(NewClass.attr, 'NewClass')
        self.assertEqual(NewClassSubclass.attr, 'NewClass')
        self.assertEqual(DeprecatedClass.attr, 'NewClass')
        self.assertEqual(DeprecatedClassSubclass.attr,
                         'DeprecatedClassSubclass')
        self.assertEqual(DeprecatedClassSubSubclass.attr,
                         'DeprecatedClassSubSubclass')

    def test_renamed_errors(self):
        class NewClass(object):
            pass

        with self.assertRaisesRegex(
                TypeError, "Declaring class 'DeprecatedClass' using the "
                "RenamedClass metaclass, but without specifying the "
                "__renamed__new_class__ class attribute"):
            class DeprecatedClass(metaclass=RenamedClass):
                __renamed_new_class__ = NewClass

        with self.assertRaisesRegex(
                TypeError, "Declaring class 'DeprecatedClass' using the "
                "RenamedClass metaclass, but without specifying the "
                "__renamed__version__ class attribute"):
            class DeprecatedClass(metaclass=RenamedClass):
                __renamed__new_class__ = NewClass

if __name__ == '__main__':
    unittest.main()
