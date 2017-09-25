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

from pyutilib.enum import Enum
from pyomo.opt.results.container import *

ProblemSense = Enum('unknown', 'minimize', 'maximize')


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

