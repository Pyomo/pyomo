#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core.base.numvalue import NumericValue

class MockFixedValue(NumericValue):
    value = 42
    def __init__(self, v = 42):
        self.value = v
    def is_fixed(self):
        return True
    def __call__(self, exception=True):
        return self.value
