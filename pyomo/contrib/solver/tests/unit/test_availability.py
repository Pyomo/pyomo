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
from pyomo.contrib.solver.common.availability import Availability, LicenseAvailability


class TestAvailability(unittest.TestCase):
    def test_statuses(self):
        self.assertTrue(bool(Availability.Available))
        self.assertFalse(bool(Availability.NotFound))
        self.assertFalse(bool(Availability.BadVersion))
        self.assertFalse(bool(Availability.NeedsCompiledExtension))

    def test_str_and_format(self):
        self.assertEqual(str(Availability.Available), "Available")
        self.assertEqual(f"{Availability.BadVersion}", "BadVersion")
        formatted = "{:>15}".format(Availability.Available)
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
