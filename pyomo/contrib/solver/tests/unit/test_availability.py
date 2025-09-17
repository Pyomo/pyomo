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

from pyomo.common import unittest
from pyomo.contrib.solver.common.availability import (
    SolverAvailability,
    LicenseAvailability,
)


class TestSolverAvailability(unittest.TestCase):
    def test_statuses(self):
        self.assertTrue(bool(SolverAvailability.Available))
        self.assertFalse(bool(SolverAvailability.NotFound))
        self.assertFalse(bool(SolverAvailability.BadVersion))
        self.assertFalse(bool(SolverAvailability.NeedsCompiledExtension))

    def test_str_and_format(self):
        self.assertEqual(str(SolverAvailability.Available), "Available")
        self.assertEqual(f"{SolverAvailability.BadVersion}", "BadVersion")
        formatted = "{:>15}".format(SolverAvailability.Available)
        self.assertIn("Available", formatted)


class TestLicenseAvailability(unittest.TestCase):
    def test_statuses(self):
        self.assertTrue(bool(LicenseAvailability.FullLicense))
        self.assertTrue(bool(LicenseAvailability.LimitedLicense))
        self.assertTrue(bool(LicenseAvailability.NotApplicable))
        self.assertFalse(bool(LicenseAvailability.NotAvailable))
        self.assertFalse(bool(LicenseAvailability.BadLicense))
        self.assertFalse(bool(LicenseAvailability.Timeout))
        self.assertFalse(bool(LicenseAvailability.Unknown))

    def test_str_and_format(self):
        self.assertEqual(str(LicenseAvailability.FullLicense), "FullLicense")
        self.assertEqual(f"{LicenseAvailability.Timeout}", "Timeout")
        formatted = "{:<20}".format(LicenseAvailability.NotApplicable)
        self.assertIn("NotApplicable", formatted)
