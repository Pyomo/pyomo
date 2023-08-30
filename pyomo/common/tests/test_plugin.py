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
import gc
from io import StringIO
import weakref

from pyomo.common.unittest import TestCase
from pyomo.common.log import LoggingIntercept
from pyomo.common.plugin_base import (
    Interface,
    Plugin,
    SingletonPlugin,
    ExtensionPoint,
    implements,
    alias,
    PluginFactory,
    PluginError,
    PluginGlobals,
    DeprecatedInterface,
)


class TestPlugin(TestCase):
    def test_plugin_interface(self):
        class IFoo(Interface):
            pass

        class myFoo(Plugin):
            implements(IFoo)

        ep = ExtensionPoint(IFoo)
        self.assertEqual(ep.extensions(), [])
        self.assertEqual(IFoo._plugins, {myFoo: {}})
        self.assertEqual(len(ep), 0)

        a = myFoo()
        self.assertEqual(ep.extensions(), [])
        self.assertEqual(IFoo._plugins, {myFoo: {0: (weakref.ref(a), False)}})
        self.assertEqual(len(ep), 0)

        a.activate()
        self.assertEqual(ep.extensions(), [a])
        self.assertEqual(IFoo._plugins, {myFoo: {0: (weakref.ref(a), True)}})
        self.assertEqual(len(ep), 1)

        a.deactivate()
        self.assertEqual(ep.extensions(), [])
        self.assertEqual(IFoo._plugins, {myFoo: {0: (weakref.ref(a), False)}})
        self.assertEqual(len(ep), 0)

        # Free a and make sure the garbage collector collects it (so
        # that the weakref will be removed from IFoo._plugins)
        a = None
        gc.collect()
        gc.collect()
        gc.collect()

        self.assertEqual(ep.extensions(), [])
        self.assertEqual(IFoo._plugins, {myFoo: {}})
        self.assertEqual(len(ep), 0)

    def test_singleton_plugin_interface(self):
        class IFoo(Interface):
            pass

        class mySingleton(SingletonPlugin):
            implements(IFoo)

        ep = ExtensionPoint(IFoo)
        self.assertEqual(ep.extensions(), [])
        self.assertEqual(
            IFoo._plugins,
            {mySingleton: {0: (weakref.ref(mySingleton.__singleton__), False)}},
        )
        self.assertIsNotNone(mySingleton.__singleton__)

        with self.assertRaisesRegex(
            RuntimeError, 'Cannot create multiple singleton plugin instances'
        ):
            mySingleton()

        class myDerivedSingleton(mySingleton):
            pass

        self.assertEqual(ep.extensions(), [])
        self.assertEqual(
            IFoo._plugins,
            {
                mySingleton: {0: (weakref.ref(mySingleton.__singleton__), False)},
                myDerivedSingleton: {
                    1: (weakref.ref(myDerivedSingleton.__singleton__), False)
                },
            },
        )
        self.assertIsNotNone(myDerivedSingleton.__singleton__)
        self.assertIsNot(mySingleton.__singleton__, myDerivedSingleton.__singleton__)

        class myDerivedNonSingleton(mySingleton):
            __singleton__ = False

        self.assertEqual(ep.extensions(), [])
        self.assertEqual(
            IFoo._plugins,
            {
                mySingleton: {0: (weakref.ref(mySingleton.__singleton__), False)},
                myDerivedSingleton: {
                    1: (weakref.ref(myDerivedSingleton.__singleton__), False)
                },
                myDerivedNonSingleton: {},
            },
        )
        self.assertIsNone(myDerivedNonSingleton.__singleton__)

        class myServiceSingleton(mySingleton):
            implements(IFoo, service=True)

        self.assertEqual(ep.extensions(), [myServiceSingleton.__singleton__])
        self.assertEqual(
            IFoo._plugins,
            {
                mySingleton: {0: (weakref.ref(mySingleton.__singleton__), False)},
                myDerivedSingleton: {
                    1: (weakref.ref(myDerivedSingleton.__singleton__), False)
                },
                myDerivedNonSingleton: {},
                myServiceSingleton: {
                    2: (weakref.ref(myServiceSingleton.__singleton__), True)
                },
            },
        )
        self.assertIsNotNone(myServiceSingleton.__singleton__)

    def test_inherit_interface(self):
        class IFoo(Interface):
            def fcn(self):
                return 'base'

            def baseFcn(self):
                return 'baseFcn'

        class myFoo(Plugin):
            implements(IFoo)

        with self.assertRaises(AttributeError):
            myFoo().fcn()

        class myFoo(Plugin):
            implements(IFoo, inherit=True)

            def fcn(self):
                return 'derived'

        self.assertEqual(myFoo().fcn(), 'derived')
        self.assertEqual(myFoo().baseFcn(), 'baseFcn')

        class IMock(Interface):
            def mock(self):
                return 'mock'

        class myCombined(myFoo):
            implements(IMock, inherit=True)

        a = myCombined()
        self.assertEqual(a.fcn(), 'derived')
        self.assertEqual(a.baseFcn(), 'baseFcn')
        self.assertEqual(a.mock(), 'mock')

    def test_plugin_factory(self):
        class IFoo(Interface):
            pass

        ep = ExtensionPoint(IFoo)

        class myFoo(Plugin):
            implements(IFoo)
            alias('my_foo', 'myFoo docs')

        factory = PluginFactory(IFoo)
        self.assertEqual(factory.services(), ['my_foo'])

        self.assertIsInstance(factory('my_foo'), myFoo)
        self.assertIsNone(factory('unknown'), None)

        self.assertEqual(factory.doc('my_foo'), 'myFoo docs')
        self.assertEqual(factory.doc('unknown'), '')

        self.assertIs(factory.get_class('my_foo'), myFoo)
        self.assertIsNone(factory.get_class('unknown'))

        a = myFoo()
        b = myFoo()

        self.assertFalse(a.enabled())
        self.assertFalse(b.enabled())
        self.assertIsNone(ep.service())

        a.activate()
        self.assertTrue(a.enabled())
        self.assertFalse(b.enabled())
        self.assertIs(ep.service(), a)

        factory.deactivate('my_foo')
        self.assertFalse(a.enabled())
        self.assertFalse(b.enabled())
        self.assertIsNone(ep.service())

        b.activate()
        self.assertFalse(a.enabled())
        self.assertTrue(b.enabled())
        self.assertIs(ep.service(), b)

        # Note: Run the GC to ensure the instance created by
        # factory('my_foo') above has been removed.
        gc.collect()
        gc.collect()
        gc.collect()

        factory.activate('my_foo')
        self.assertTrue(a.enabled())
        self.assertTrue(b.enabled())
        with self.assertRaisesRegex(
            PluginError,
            r"The ExtensionPoint does not have a unique service!  "
            r"2 services are defined for interface 'IFoo' \(key=None\).",
        ):
            self.assertIsNone(ep.service())

        a.deactivate()
        self.assertFalse(a.enabled())
        self.assertTrue(b.enabled())
        self.assertIs(ep.service(), b)

        factory.activate('unknown')
        self.assertFalse(a.enabled())
        self.assertTrue(b.enabled())
        self.assertIs(ep.service(), b)

        factory.deactivate('unknown')
        self.assertFalse(a.enabled())
        self.assertTrue(b.enabled())
        self.assertIs(ep.service(), b)

    def test_deprecation(self):
        out = StringIO()
        with LoggingIntercept(out):
            PluginGlobals.add_env(None)
        self.assertIn(
            "Pyomo only supports a single global environment",
            out.getvalue().replace('\n', ' '),
        )

        out = StringIO()
        with LoggingIntercept(out):
            PluginGlobals.pop_env()
        self.assertIn(
            "Pyomo only supports a single global environment",
            out.getvalue().replace('\n', ' '),
        )

        out = StringIO()
        with LoggingIntercept(out):
            PluginGlobals.clear()
        self.assertIn(
            "Pyomo only supports a single global environment",
            out.getvalue().replace('\n', ' '),
        )

        class IFoo(Interface):
            pass

        out = StringIO()
        with LoggingIntercept(out):

            class myFoo(Plugin):
                alias('myFoo', subclass=True)

        self.assertIn(
            "alias() function does not support the subclass",
            out.getvalue().replace('\n', ' '),
        )

        out = StringIO()
        with LoggingIntercept(out):

            class myFoo(Plugin):
                implements(IFoo, namespace='here')

        self.assertIn(
            "only supports a single global namespace.",
            out.getvalue().replace('\n', ' '),
        )

        class IGone(DeprecatedInterface):
            __deprecated_version__ = '1.2.3'
            pass

        out = StringIO()
        with LoggingIntercept(out):

            class myFoo(Plugin):
                implements(IGone)

        self.assertIn(
            "The IGone interface has been deprecated", out.getvalue().replace('\n', ' ')
        )

        out = StringIO()
        with LoggingIntercept(out):
            ExtensionPoint(IGone).extensions()
        self.assertIn(
            "The IGone interface has been deprecated", out.getvalue().replace('\n', ' ')
        )

        class ICustomGone(DeprecatedInterface):
            __deprecated_message__ = 'This interface is gone!'
            __deprecated_version__ = '1.2.3'

        out = StringIO()
        with LoggingIntercept(out):

            class myFoo(Plugin):
                implements(ICustomGone)

        self.assertIn("This interface is gone!", out.getvalue().replace('\n', ' '))

        out = StringIO()
        with LoggingIntercept(out):
            ExtensionPoint(ICustomGone).extensions()
        self.assertIn("This interface is gone!", out.getvalue().replace('\n', ' '))
