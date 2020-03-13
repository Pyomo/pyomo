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
import pyutilib.th as unittest

from pyomo.common.dependencies import (
    attempt_import, ModuleUnavailable, DeferredImportModule,
    DeferredImportIndicator, DeferredImportError,
)

import pyomo.common.tests.dep_mod as dep_mod
from pyomo.common.tests.dep_mod import (
    numpy, numpy_available,
    bogus_nonexisting_module as bogus_nem,
    bogus_nonexisting_module_available as has_bogus_nem,
)

bogus_nonexisting_module, bogus_nonexisting_module_available \
    = attempt_import('bogus_nonexisting_module', defer_check=True)

class TestDependencies(unittest.TestCase):
    def test_local_deferred_import(self):
        self.assertIs(type(bogus_nonexisting_module_available),
                      DeferredImportIndicator)
        self.assertIs(type(bogus_nonexisting_module), DeferredImportModule)
        if bogus_nonexisting_module_available:
            self.fail("Casting bogus_nonexisting_module_available "
                      "to bool returned True")
        self.assertIs(bogus_nonexisting_module_available, False)
        self.assertIs(type(bogus_nonexisting_module), ModuleUnavailable)

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
        self.assertIs(type(dep_mod.bogus_nonexisting_module),
                      ModuleUnavailable)
        
    def test_min_version(self):
        mod, avail = attempt_import('pyomo.common.tests.dep_mod',
                                    minimum_version='1.0',
                                    defer_check=False)
        self.assertTrue(avail)
        self.assertTrue(inspect.ismodule(mod))

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
