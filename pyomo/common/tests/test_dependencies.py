#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import inspect
from six import StringIO

import pyutilib.th as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
    attempt_import, ModuleUnavailable, DeferredImportModule,
    DeferredImportIndicator, DeferredImportError,
    _DeferredAnd, _DeferredOr, check_min_version
)

import pyomo.common.tests.dep_mod as dep_mod
from pyomo.common.tests.dep_mod import (
    bogus_nonexisting_module as bogus_nem,
    bogus_nonexisting_module_available as has_bogus_nem,
)

bogus, bogus_available \
    = attempt_import('nonexisting.module.bogus', defer_check=True)

# global objects for the submodule tests
def _finalize_pyo(module, available):
    if available:
        import pyomo.core

pyo, pyo_available = attempt_import(
    'pyomo', alt_names=['pyo'],
    deferred_submodules={'version': None,
                         'common.tests.dep_mod': ['dm']})
dm = pyo.common.tests.dep_mod

class TestDependencies(unittest.TestCase):
    def test_import_error(self):
        module_obj, module_available = attempt_import(
            '__there_is_no_module_named_this__',
            'Testing import of a non-existant module',
            defer_check=False)
        self.assertFalse(module_available)
        with self.assertRaisesRegex(
                DeferredImportError, 'Testing import of a non-existant module'):
            module_obj.try_to_call_a_method()

        # Note that some attribute will intentionally raise
        # AttributeErrors and NOT DeferredImportError:
        with self.assertRaisesRegex(
                AttributeError, "'ModuleUnavailable' object has no "
                "attribute '__sphinx_mock__'"):
            module_obj.__sphinx_mock__

                
    def test_import_success(self):
        module_obj, module_available = attempt_import(
            'pyutilib','Testing import of PyUtilib', defer_check=False)
        self.assertTrue(module_available)
        import pyutilib
        self.assertTrue(module_obj is pyutilib)

    def test_local_deferred_import(self):
        self.assertIs(type(bogus_available), DeferredImportIndicator)
        self.assertIs(type(bogus), DeferredImportModule)
        if bogus_available:
            self.fail("Casting bogus_available to bool returned True")
        self.assertIs(bogus_available, False)
        # Note: this also tests the implicit alt_names for dotted imports
        self.assertIs(type(bogus), ModuleUnavailable)
        with self.assertRaisesRegexp(
                DeferredImportError, "The nonexisting.module.bogus module "
                "\(an optional Pyomo dependency\) failed to import"):
            bogus.hello

    def test_imported_deferred_import(self):
        self.assertIs(type(has_bogus_nem), DeferredImportIndicator)
        self.assertIs(type(bogus_nem), DeferredImportModule)
        with self.assertRaisesRegexp(
                DeferredImportError, "The bogus_nonexisting_module module "
                "\(an optional Pyomo dependency\) failed to import"):
            bogus_nem.hello
        self.assertIs(has_bogus_nem, False)
        self.assertIs(type(bogus_nem), ModuleUnavailable)
        self.assertIs(dep_mod.bogus_nonexisting_module_available, False)
        self.assertIs(type(dep_mod.bogus_nonexisting_module), ModuleUnavailable)

    def test_min_version(self):
        mod, avail = attempt_import('pyomo.common.tests.dep_mod',
                                    minimum_version='1.0',
                                    defer_check=False)
        self.assertTrue(avail)
        self.assertTrue(inspect.ismodule(mod))
        self.assertTrue(check_min_version(mod, '1.0'))
        self.assertFalse(check_min_version(mod, '2.0'))

        mod, avail = attempt_import('pyomo.common.tests.dep_mod',
                                    minimum_version='2.0',
                                    defer_check=False)
        self.assertFalse(avail)
        self.assertIs(type(mod), ModuleUnavailable)
        with self.assertRaisesRegex(
                DeferredImportError, "The pyomo.common.tests.dep_mod module "
                "version 1.5 does not satisfy the minimum version 2.0"):
            mod.hello

        mod, avail = attempt_import('pyomo.common.tests.dep_mod',
                                    error_message="Failed import",
                                    minimum_version='2.0',
                                    defer_check=False)
        self.assertFalse(avail)
        self.assertIs(type(mod), ModuleUnavailable)
        with self.assertRaisesRegex(
                DeferredImportError, "Failed import "
                "\(version 1.5 does not satisfy the minimum version 2.0\)"):
            mod.hello

        # Verify check_min_version works with deferred imports

        mod, avail = attempt_import('pyomo.common.tests.dep_mod',
                                    defer_check=True)
        self.assertTrue(check_min_version(mod, '1.0'))

        mod, avail = attempt_import('pyomo.common.tests.dep_mod',
                                    defer_check=True)
        self.assertFalse(check_min_version(mod, '2.0'))

    def test_and_or(self):
        mod0, avail0 = attempt_import('pyutilib',
                                      defer_check=True)
        mod1, avail1 = attempt_import('pyomo.common.tests.dep_mod',
                                      defer_check=True)
        mod2, avail2 = attempt_import('pyomo.common.tests.dep_mod',
                                      minimum_version='2.0',
                                      defer_check=True)

        _and = avail0 & avail1
        self.assertIsInstance(_and, _DeferredAnd)

        _or = avail1 | avail2
        self.assertIsInstance(_or, _DeferredOr)

        # Nothing has been resolved yet
        self.assertIsNone(avail0._available)
        self.assertIsNone(avail1._available)
        self.assertIsNone(avail2._available)

        # Shortcut boolean evaluation only partially resolves things
        self.assertTrue(_or)
        self.assertIsNone(avail0._available)
        self.assertTrue(avail1._available)
        self.assertIsNone(avail2._available)

        self.assertTrue(_and)
        self.assertTrue(avail0._available)
        self.assertTrue(avail1._available)
        self.assertIsNone(avail2._available)

        # Testing compound operations
        _and_and = avail0 & avail1 & avail2
        self.assertFalse(_and_and)

        _and_or = avail0 & avail1 | avail2
        self.assertTrue(_and_or)

        # Verify operator prescedence
        _or_and = avail0 | avail2 & avail2
        self.assertTrue(_or_and)
        _or_and = (avail0 | avail2) & avail2
        self.assertFalse(_or_and)

        _or_or = avail0 | avail1 | avail2
        self.assertTrue(_or_or)

        # Verify rand / ror
        _rand = True & avail1
        self.assertIsInstance(_rand, _DeferredAnd)
        self.assertTrue(_rand)

        _ror = False | avail1
        self.assertIsInstance(_ror, _DeferredOr)
        self.assertTrue(_ror)


    def test_callbacks(self):
        ans = []
        def _record_avail(module, avail):
            ans.append(avail)

        mod0, avail0 = attempt_import('pyutilib',
                                      defer_check=True,
                                      callback=_record_avail)
        mod1, avail1 = attempt_import('pyomo.common.tests.dep_mod',
                                      minimum_version='2.0',
                                      defer_check=True,
                                      callback=_record_avail)

        self.assertEqual(ans, [])
        self.assertTrue(avail0)
        self.assertEqual(ans, [True])
        self.assertFalse(avail1)
        self.assertEqual(ans, [True,False])

    def test_import_exceptions(self):
        mod, avail = attempt_import('pyomo.common.tests.dep_mod_except',
                                    defer_check=True)
        with self.assertRaisesRegex(ValueError, "cannot import module"):
            bool(avail)
        # second test will not re-trigger the exception
        self.assertFalse(avail)

        mod, avail = attempt_import('pyomo.common.tests.dep_mod_except',
                                    defer_check=True,
                                    only_catch_importerror=False)
        self.assertFalse(avail)
        self.assertFalse(avail)

    def test_generate_warning(self):
        mod, avail = attempt_import('pyomo.common.tests.dep_mod_except',
                                    defer_check=True,
                                    only_catch_importerror=False)

        # Test generate warning
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.common'):
            mod.generate_import_warning()
        self.assertEqual(
            log.getvalue(), "The pyomo.common.tests.dep_mod_except module "
            "(an optional Pyomo dependency) failed to import\n")

        log = StringIO()
        with LoggingIntercept(log, 'pyomo.core.base'):
            mod.generate_import_warning('pyomo.core.base')
        self.assertEqual(
            log.getvalue(), "The pyomo.common.tests.dep_mod_except module "
            "(an optional Pyomo dependency) failed to import\n")

    def test_importer(self):
        attempted_import = []
        def _importer():
            attempted_import.append(True)
            return attempt_import('pyomo.common.tests.dep_mod',
                                  defer_check=False)[0]

        mod, avail = attempt_import('foo',
                                    importer=_importer,
                                    defer_check=True)

        self.assertEqual(attempted_import, [])
        self.assertIsInstance(mod, DeferredImportModule)
        self.assertTrue(avail)
        self.assertEqual(attempted_import, [True])
        self.assertIs(mod._indicator_flag._module, dep_mod)

    def test_deferred_submodules(self):
        import pyomo
        pyo_ver = pyomo.version.version

        self.assertIsInstance(pyo, DeferredImportModule)
        self.assertIsNone(pyo._submodule_name)
        self.assertEqual(pyo_available._deferred_submodules,
                         {'.version': None,
                          '.common': None,
                          '.common.tests': None,
                          '.common.tests.dep_mod': ['dm']})
        # This doesn't cause test_mod to be resolved
        version = pyo.version
        self.assertIsInstance(pyo, DeferredImportModule)
        self.assertIsNone(pyo._submodule_name)
        self.assertIsInstance(dm, DeferredImportModule)
        self.assertEqual(dm._submodule_name, '.common.tests.dep_mod')
        self.assertIsInstance(version, DeferredImportModule)
        self.assertEqual(version._submodule_name, '.version')
        # This causes the global objects to be resolved
        self.assertEqual(version.version, pyo_ver)
        self.assertTrue(inspect.ismodule(pyo))
        self.assertTrue(inspect.ismodule(dm))

        with self.assertRaisesRegex(
                ValueError,
                "deferred_submodules is only valid if defer_check==True"):
            mod, mod_available \
                = attempt_import('nonexisting.module', defer_check=False,
                                 deferred_submodules={'submod': None})

        mod, mod_available \
            = attempt_import('nonexisting.module', defer_check=True,
                             deferred_submodules={'submod.subsubmod': None})
        self.assertIs(type(mod), DeferredImportModule)
        self.assertFalse(mod_available)
        _mod = mod_available._module
        self.assertIs(type(_mod), ModuleUnavailable)
        self.assertTrue(hasattr(_mod, 'submod'))
        self.assertIs(type(_mod.submod), ModuleUnavailable)
        self.assertTrue(hasattr(_mod.submod, 'subsubmod'))
        self.assertIs(type(_mod.submod.subsubmod), ModuleUnavailable)

if __name__ == '__main__':
    unittest.main()
