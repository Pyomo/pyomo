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

from typing import Optional

from pyomo.common.dependencies import attempt_import

knitro, KNITRO_AVAILABLE = attempt_import("knitro")


def get_version() -> Optional[str]:
    if not KNITRO_AVAILABLE:
        return None
    return knitro.__version__
