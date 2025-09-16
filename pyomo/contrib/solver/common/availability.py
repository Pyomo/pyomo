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

from pyomo.common.enums import IntEnum


class Availability(IntEnum):
    """
    Class to capture different statuses in which a solver can exist in
    order to record its availability for use.
    """

    Available = 1
    NotFound = 0
    BadVersion = -1
    NeedsCompiledExtension = -2

    def __bool__(self):
        return self._value_ > 0

    def __format__(self, format_spec):
        return format(self.name, format_spec)

    def __str__(self):
        return self.name


class LicenseAvailability(IntEnum):
    """
    Runtime status for licensing. Independent from
    overall solver availability. A return value > 0 is "usable in some form".
    """

    FullLicense = 3
    LimitedLicense = 2
    NotApplicable = 1
    NotAvailable = 0
    BadLicense = -1
    Timeout = -2
    Unknown = -3

    def __bool__(self):
        return self._value_ > 0

    def __format__(self, format_spec):
        return format(self.name, format_spec)

    def __str__(self):
        return self.name
