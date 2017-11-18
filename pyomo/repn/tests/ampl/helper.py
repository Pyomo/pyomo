#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr import NumericValue


class MockFixedValue(NumericValue):
    value = 42
    def __init__(self, v = 42):
        self.value = v
    def is_fixed(self):
        return True
    def __call__(self, exception=True):
        return self.value
