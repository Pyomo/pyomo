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
from .doe import DesignOfExperiments, ObjectiveLib, FiniteDifferenceStep
from .utils import rescale_FIM

# Deprecation errors for old Pyomo.DoE interface classes and structures
from pyomo.common.deprecation import deprecated

deprecation_message = (
    "Pyomo.DoE has been refactored. The current interface utilizes Experiment "
    "objects that label unknown parameters, experiment inputs, experiment outputs "
    "and measurement error. This avoids string-based naming which is fragile. For "
    "instructions to use the new interface, please see the Pyomo.DoE under the contributed "
    "packages documentation at `https://pyomo.readthedocs.io/en/latest/contributed_packages/doe/doe.html`"
)


@deprecated(
    "Use of MeasurementVariables in Pyomo.DoE is no longer supported.", version='6.8.0'
)
class MeasurementVariables:
    def __init__(self, *args):
        raise RuntimeError(deprecation_message)


@deprecated(
    "Use of DesignVariables in Pyomo.DoE is no longer supported.", version='6.8.0'
)
class DesignVariables:
    def __init__(self, *args):
        raise RuntimeError(deprecation_message)


@deprecated(
    "Use of ModelOptionLib in Pyomo.DoE is no longer supported.", version='6.8.0'
)
class ModelOptionLib:
    def __init__(self, *args):
        raise RuntimeError(deprecation_message)
