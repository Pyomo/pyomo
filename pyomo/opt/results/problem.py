#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['ProblemInformation', 'ProblemSense']

import enum
from pyomo.opt.results.container import MapContainer

class ProblemSense(str, enum.Enum):
    unknown='unknown'
    minimize='minimize'
    maximize='maximize'

    # Overloading __str__ is needed to match the behavior of the old
    # pyutilib.enum class (removed June 2020). There are spots in the
    # code base that expect the string representation for items in the
    # enum to not include the class name. New uses of enum shouldn't
    # need to do this.
    def __str__(self):
        return self.value


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
        self.declare('sense', value=ProblemSense.unknown, required=True)

