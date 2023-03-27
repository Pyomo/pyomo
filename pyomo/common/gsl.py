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

import logging
import platform
from pyomo.common import Library
from pyomo.common.deprecation import deprecated

logger = logging.getLogger('pyomo.common')


@deprecated(
    "Use of get_gsl is deprecated and NO LONGER FUNCTIONS as of February 9, 2023. ",
    version='6.5.0',
)
def get_gsl(downloader):
    logger.info(
        "As of February 9, 2023, AMPL GSL can no longer be downloaded\
        through download-extensions. Visit https://portal.ampl.com/\
        to download the AMPL GSL binaries."
    )


def find_GSL():
    # FIXME: the GSL interface is currently broken in PyPy:
    if platform.python_implementation().lower().startswith('pypy'):
        return None
    return Library('amplgsl.dll').path()
