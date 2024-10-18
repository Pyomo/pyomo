#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from . import common
from .version import __version__


from pyomo.common.deprecation import moved_module

moved_module(
    'pyomo.pysp',
    'pysp',
    version='6.0',
    msg="PySP has been removed from the pyomo.pysp namespace.  "
    "Beginning in Pyomo 6.0, PySP is distributed as a separate "
    "package.  Please see https://github.com/Pyomo/pysp for "
    "information on downloading and installing PySP",
)
del moved_module
