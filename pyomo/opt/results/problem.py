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

from pyomo.common.enums import ExtendedEnumType, IntEnum, ObjectiveSense
from pyomo.opt.results.container import MapContainer


class ProblemSense(IntEnum, metaclass=ExtendedEnumType):
    __base_enum__ = ObjectiveSense

    unknown = 0

    def __str__(self):
        return self.name


class ProblemInformation(MapContainer):
    def __init__(self):
        super().__init__()
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
        self.declare('sense', value=ProblemSense.unknown, required=True)
