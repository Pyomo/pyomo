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

import enum
from pyomo.opt.results.container import MapContainer

from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.enums import ObjectiveSense


class ProblemSenseType(type):
    @deprecated(
        "pyomo.opt.results.problem.ProblemSense has been replaced by "
        "pyomo.common.enums.ObjectiveSense",
        version="6.7.2.dev0",
    )
    def __getattr__(cls, attr):
        if attr == 'minimize':
            return ObjectiveSense.minimize
        if attr == 'maximize':
            return ObjectiveSense.maximize
        if attr == 'unknown':
            deprecation_warning(
                "ProblemSense.unknown is no longer an allowable option.  "
                "Mapping 'unknown' to 'minimize'",
                version="6.7.2.dev0",
            )
            return ObjectiveSense.minimize
        raise AttributeError(attr)


class ProblemSense(metaclass=ProblemSenseType):
    pass


class ProblemInformation(MapContainer):
    def __init__(self):
        MapContainer.__init__(self)
        self.declare('name')
        self.declare('lower_bound', value=float('-inf'))
        self.declare('upper_bound', value=float('inf'))
        self.declare('number_of_objectives', value=1, required=True)
        self.declare('number_of_constraints', value=0)
        self.declare('number_of_variables', value=0)
        self.declare('number_of_binary_variables')
        self.declare('number_of_integer_variables')
        self.declare('number_of_continuous_variables')
        self.declare('number_of_nonzeros')
        self.declare('sense', value=ProblemSense.minimize, required=True)
