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

import inspect
from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import (
    attempt_import,
    ModuleUnavailable,
    DeferredImportModule,
    DeferredImportIndicator,
    DeferredImportError,
    UnavailableClass,
    _DeferredAnd,
    _DeferredOr,
    check_min_version,
    dill,
    dill_available,
)

import pyomo.common.tests.dep_mod as dep_mod

from . import deps


# global objects for the submodule tests
def _finalize_pyo(module, available):
    if available:
        import pyomo.core


class TestDependencies(unittest.TestCase):
    def test_import_error(self):
        module_obj, module_available = attempt_import(
            '__there_is_no_module_named_this__',
            'Testing import of a non-existent module',
            defer_check=False,
        )
        self.assertFalse(module_available)
        with self.assertRaisesRegex(
            DeferredImportError, 'Testing import of a non-existent module'
        ):
            module_obj.try_to_call_a_method()

        # Note that some attribute will intentionally raise
        # AttributeErrors and NOT DeferredImportError:
        with self.assertRaisesRegex(
            AttributeError,
            "'ModuleUnavailable' object has no attribute '__sphinx_mock__'",
        ):
            module_obj.__sphinx_mock__

    @unittest.skipUnless(dill_available, "Test requires dill module")
    def test_pickle(self):
        self.assertIs(deps.pkl_test.__class__, DeferredImportModule)
        # Pickle the DeferredImportModule class
        pkl = dill.dumps(deps.pkl_test)
        deps.new_pkl_test = dill.loads(pkl)
        self.assertIs(deps.pkl_test.__class__, deps.new_pkl_test.__class__)
        self.assertIs(deps.new_pkl_test.__class__, DeferredImportModule)
        self.assertIsNot(deps.pkl_test, deps.new_pkl_test)
        self.assertIn('submod', deps.new_pkl_test.__dict__)
        with self.assertRaisesRegex(
            DeferredImportError, 'nonexisting.module.pickle_test module'
        ):
            deps.new_pkl_test.try_to_call_a_method()
        # Pickle the ModuleUnavailable class
        self.assertIs(deps.new_pkl_test.__class__, ModuleUnavailable)
        pkl = dill.dumps(deps.new_pkl_test)
        new_pkl_test_2 = dill.loads(pkl)
        self.assertIs(deps.new_pkl_test.__class__, new_pkl_test_2.__class__)
        self.assertIsNot(deps.new_pkl_test, new_pkl_test_2)
        self.assertIs(new_pkl_test_2.__class__, ModuleUnavailable)

    def test_import_success(self):
        module_obj, module_available = attempt_import(
            'ply', 'Testing import of ply', defer_check=False
        )
        self.assertTrue(module_available)
        import ply

        self.assertTrue(module_obj is ply)

    def test_local_deferred_import(self):
        self.assertIs(type(deps.bogus_available), DeferredImportIndicator)
        self.assertIs(type(deps.bogus), DeferredImportModule)
        if deps.bogus_available:
            self.fail("Casting bogus_available to bool returned True")
        self.assertIs(deps.bogus_available, False)
        # Note: this also tests the implicit alt_names for dotted imports
        self.assertIs(type(deps.bogus), ModuleUnavailable)
        with self.assertRaisesRegex(
            DeferredImportError,
            "The nonexisting.module.bogus module "
            r"\(an optional Pyomo dependency\) failed to import",
        ):
            deps.bogus.hello

    def test_imported_deferred_import(self):
        self.assertIs(type(deps.has_bogus_nem), DeferredImportIndicator)
        self.assertIs(type(deps.bogus_nem), DeferredImportModule)
        with self.assertRaisesRegex(
            DeferredImportError,
            "The bogus_nonexisting_module module "
            r"\(an optional Pyomo dependency\) failed to import",
        ):
            deps.test_access_bogus_hello()
        self.assertIs(deps.has_bogus_nem, False)
        self.assertIs(type(deps.bogus_nem), ModuleUnavailable)
        self.assertIs(dep_mod.bogus_nonexisting_module_available, False)
        self.assertIs(type(dep_mod.bogus_nonexisting_module), ModuleUnavailable)

    def test_min_version(self):
        mod, avail = attempt_import(
            'pyomo.common.tests.dep_mod', minimum_version='1.0', defer_check=False
        )
        self.assertTrue(avail)
        self.assertTrue(inspect.ismodule(mod))
        self.assertTrue(check_min_version(mod, '1.0'))
        self.assertFalse(check_min_version(mod, '2.0'))

        mod, avail = attempt_import(
            'pyomo.common.tests.dep_mod', minimum_version='2.0', defer_check=False
        )
        self.assertFalse(avail)
        self.assertIs(type(mod), ModuleUnavailable)
        with self.assertRaisesRegex(
            DeferredImportError,
            "The pyomo.common.tests.dep_mod module "
            "version 1.5 does not satisfy the minimum version 2.0",
        ):
            mod.hello

        mod, avail = attempt_import(
            'pyomo.common.tests.dep_mod',
            error_message="Failed import",
            minimum_version='2.0',
            defer_check=False,
        )
        self.assertFalse(avail)
        self.assertIs(type(mod), ModuleUnavailable)
        with self.assertRaisesRegex(
            DeferredImportError,
            "Failed import "
            r"\(version 1.5 does not satisfy the minimum version 2.0\)",
        ):
            mod.hello

        # Verify check_min_version works with deferred imports

        mod, avail = attempt_import('pyomo.common.tests.dep_mod', defer_check=True)
        self.assertTrue(check_min_version(mod, '1.0'))

        mod, avail = attempt_import('pyomo.common.tests.dep_mod', defer_check=True)
        self.assertFalse(check_min_version(mod, '2.0'))

        # Verify check_min_version works when called directly

        mod, avail = attempt_import('pyomo.common.tests.dep_mod', minimum_version='1.0')
        self.assertTrue(check_min_version(mod, '1.0'))

        mod, avail = attempt_import('pyomo.common.tests.bogus', minimum_version='1.0')
        self.assertFalse(check_min_version(mod, '1.0'))

    def test_and_or(self):
        mod0, avail0 = attempt_import('ply', defer_check=True)
        mod1, avail1 = attempt_import('pyomo.common.tests.dep_mod', defer_check=True)
        mod2, avail2 = attempt_import(
            'pyomo.common.tests.dep_mod', minimum_version='2.0', defer_check=True
        )

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

        mod0, avail0 = attempt_import('ply', defer_check=True, callback=_record_avail)
        mod1, avail1 = attempt_import(
            'pyomo.common.tests.dep_mod',
            minimum_version='2.0',
            defer_check=True,
            callback=_record_avail,
        )

        self.assertEqual(ans, [])
        self.assertTrue(avail0)
        self.assertEqual(ans, [True])
        self.assertFalse(avail1)
        self.assertEqual(ans, [True, False])

    def test_import_exceptions(self):
        mod, avail = attempt_import(
            'pyomo.common.tests.dep_mod_except',
            defer_check=True,
            only_catch_importerror=True,
        )
        with self.assertRaisesRegex(ValueError, "cannot import module"):
            bool(avail)
        # second test will not re-trigger the exception
        self.assertFalse(avail)

        mod, avail = attempt_import(
            'pyomo.common.tests.dep_mod_except',
            defer_check=True,
            only_catch_importerror=False,
        )
        self.assertFalse(avail)
        self.assertFalse(avail)

        mod, avail = attempt_import(
            'pyomo.common.tests.dep_mod_except',
            defer_check=True,
            catch_exceptions=(ImportError, ValueError),
        )
        self.assertFalse(avail)
        self.assertFalse(avail)

        with self.assertRaisesRegex(
            ValueError,
            'Cannot specify both only_catch_importerror and catch_exceptions',
        ):
            mod, avail = attempt_import(
                'pyomo.common.tests.dep_mod_except',
                defer_check=True,
                only_catch_importerror=True,
                catch_exceptions=(ImportError,),
            )

    def test_generate_warning(self):
        mod, avail = attempt_import(
            'pyomo.common.tests.dep_mod_except',
            defer_check=True,
            only_catch_importerror=False,
        )

        # Test generate warning
        log = StringIO()
        dep = StringIO()
        with LoggingIntercept(dep, 'pyomo.common.tests'):
            with LoggingIntercept(log, 'pyomo.common'):
                mod.generate_import_warning()
        self.assertIn(
            "The pyomo.common.tests.dep_mod_except module "
            "(an optional Pyomo dependency) failed to import",
            log.getvalue(),
        )
        self.assertIn(
            "DEPRECATED: use :py:class:`log_import_warning()`", dep.getvalue()
        )

        log = StringIO()
        dep = StringIO()
        with LoggingIntercept(dep, 'pyomo'):
            with LoggingIntercept(log, 'pyomo.core.base'):
                mod.generate_import_warning('pyomo.core.base')
        self.assertIn(
            "The pyomo.common.tests.dep_mod_except module "
            "(an optional Pyomo dependency) failed to import",
            log.getvalue(),
        )
        self.assertIn(
            "DEPRECATED: use :py:class:`log_import_warning()`", dep.getvalue()
        )

    def test_log_warning(self):
        mod, avail = attempt_import(
            'pyomo.common.tests.dep_mod_except',
            defer_check=True,
            only_catch_importerror=False,
        )
        log = StringIO()
        dep = StringIO()
        with LoggingIntercept(dep, 'pyomo'):
            with LoggingIntercept(log, 'pyomo.common'):
                mod.log_import_warning()
        self.assertIn(
            "The pyomo.common.tests.dep_mod_except module "
            "(an optional Pyomo dependency) failed to import",
            dep.getvalue(),
        )
        self.assertNotIn("DEPRECATED:", dep.getvalue())
        self.assertEqual("", log.getvalue())

        log = StringIO()
        dep = StringIO()
        with LoggingIntercept(dep, 'pyomo'):
            with LoggingIntercept(log, 'pyomo.core.base'):
                mod.log_import_warning('pyomo.core.base')
        self.assertIn(
            "The pyomo.common.tests.dep_mod_except module "
            "(an optional Pyomo dependency) failed to import",
            log.getvalue(),
        )
        self.assertEqual("", dep.getvalue())

        log = StringIO()
        with LoggingIntercept(dep, 'pyomo'):
            with LoggingIntercept(log, 'pyomo.core.base'):
                mod.log_import_warning('pyomo.core.base', "Custom")
        self.assertIn(
            "Custom (import raised ValueError: cannot import module)", log.getvalue()
        )
        self.assertEqual("", dep.getvalue())

    def test_importer(self):
        attempted_import = []

        def _importer():
            attempted_import.append(True)
            return attempt_import('pyomo.common.tests.dep_mod', defer_check=False)[0]

        mod, avail = attempt_import('foo', importer=_importer, defer_check=True)

        self.assertEqual(attempted_import, [])
        self.assertIsInstance(mod, DeferredImportModule)
        self.assertTrue(avail)
        self.assertEqual(attempted_import, [True])
        self.assertIs(mod._indicator_flag._module, dep_mod)

    def test_deferred_submodules(self):
        import pyomo

        pyo_ver = pyomo.version.version

        self.assertIsInstance(deps.pyo, DeferredImportModule)
        self.assertIsNone(deps.pyo._submodule_name)
        self.assertEqual(
            deps.pyo_available._deferred_submodules,
            ['.version', '.common', '.common.tests', '.common.tests.dep_mod'],
        )
        # This doesn't cause test_mod to be resolved
        version = deps.pyo.version
        self.assertIsInstance(deps.pyo, DeferredImportModule)
        self.assertIsNone(deps.pyo._submodule_name)
        self.assertIsInstance(deps.dm, DeferredImportModule)
        self.assertEqual(deps.dm._submodule_name, '.common.tests.dep_mod')
        self.assertIsInstance(version, DeferredImportModule)
        self.assertEqual(version._submodule_name, '.version')
        # This causes the global objects to be resolved
        self.assertEqual(version.version, pyo_ver)
        self.assertTrue(inspect.ismodule(deps.pyo))
        self.assertTrue(inspect.ismodule(deps.dm))

        with self.assertRaisesRegex(
            ValueError, "deferred_submodules is only valid if defer_check==True"
        ):
            mod, mod_available = attempt_import(
                'nonexisting.module',
                defer_check=False,
                deferred_submodules={'submod': None},
            )

        mod, mod_available = attempt_import(
            'nonexisting.module',
            defer_check=True,
            deferred_submodules={'submod.subsubmod': None},
        )
        self.assertIs(type(mod), DeferredImportModule)
        self.assertFalse(mod_available)
        _mod = mod_available._module
        self.assertIs(type(_mod), ModuleUnavailable)
        self.assertTrue(hasattr(_mod, 'submod'))
        self.assertIs(type(_mod.submod), ModuleUnavailable)
        self.assertTrue(hasattr(_mod.submod, 'subsubmod'))
        self.assertIs(type(_mod.submod.subsubmod), ModuleUnavailable)

    def test_UnavailableClass(self):
        module_obj, module_available = attempt_import(
            '__there_is_no_module_named_this__',
            'Testing import of a non-existent module',
            defer_check=False,
        )

        class A_Class(UnavailableClass(module_obj)):
            pass

        with self.assertRaisesRegex(
            DeferredImportError,
            "The class 'A_Class' cannot be created because a needed optional "
            r"dependency was not found \(import raised ModuleNotFoundError: No "
            r"module named '__there_is_no_module_named_this__'\)",
        ):
            A_Class()

        with self.assertRaisesRegex(
            DeferredImportError,
            "The class attribute 'A_Class.method' is not available because a "
            r"needed optional dependency was not found \(import raised "
            "ModuleNotFoundError: No module named "
            r"'__there_is_no_module_named_this__'\)",
        ):
            A_Class.method()


if __name__ == '__main__':
    unittest.main()
