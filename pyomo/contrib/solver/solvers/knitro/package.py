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

import io
from typing import Optional

from pyomo.common.tee import TeeStream, capture_output
from pyomo.contrib.solver.common.base import Availability
from pyomo.contrib.solver.solvers.knitro.api import (
    KNITRO_AVAILABLE,
    get_version,
    knitro,
)


class Package:
    """Manages the global KNITRO license context and provides utility methods for license handling.

    This class handles license initialization, release, context creation, version reporting,
    and license availability checks for the KNITRO solver.
    """

    _license_context = None

    @staticmethod
    def initialize_license():
        """Initialize the global KNITRO license context if not already initialized.

        Returns:
            The KNITRO license context object.

        """
        if Package._license_context is None:
            Package._license_context = knitro.KN_checkout_license()
        return Package._license_context

    @staticmethod
    def release_license() -> None:
        """Release the global KNITRO license context if it exists."""
        if Package._license_context is not None:
            knitro.KN_release_license(Package._license_context)
            Package._license_context = None

    @staticmethod
    def create_context():
        """Create a new KNITRO context using the global license context.

        Returns:
            The new KNITRO context object.

        """
        lmc = Package.initialize_license()
        return knitro.KN_new_lm(lmc)

    @staticmethod
    def get_version() -> Optional[tuple[int, int, int]]:
        """Get the version of the KNITRO solver as a tuple.

        Returns:
            tuple[int, int, int]: The (major, minor, patch) version of KNITRO
            or None if KNITRO version could not be determined.

        """
        version = get_version()
        if version is None:
            return None
        major, minor, patch = map(int, version.split("."))
        return major, minor, patch

    @staticmethod
    def check_availability() -> Availability:
        """Check if the KNITRO solver and license are available.

        Returns:
            Availability: The availability status (FullLicense, BadLicense, NotFound).

        """
        if not bool(KNITRO_AVAILABLE):
            return Availability.NotFound
        try:
            stream = io.StringIO()
            with capture_output(TeeStream(stream), capture_fd=True):
                kc = Package.create_context()
                knitro.KN_free(kc)
            # TODO: parse the stream to check the license type.
            return Availability.FullLicense
        except Exception:
            return Availability.BadLicense


class PackageChecker:
    _available_cache: Optional[Availability]

    def __init__(self) -> None:
        self._available_cache = None

    def available(self) -> Availability:
        if self._available_cache is None:
            self._available_cache = Package.check_availability()
        return self._available_cache

    def version(self) -> Optional[tuple[int, int, int]]:
        return Package.get_version()
