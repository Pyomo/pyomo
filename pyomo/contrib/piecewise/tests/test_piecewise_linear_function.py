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

import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.environ import ConcreteModel, log, Var

class TestPiecewiseLinearFunction(unittest.TestCase):
    def test_pw_linear_approx_of_ln_x(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        m.f = log(m.x)

        m.pw = PiecewiseLinearFunction(points=[1, 3, 6, 10], function=m.f)

    def test_pw_linear_approx_of_paraboloid(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 9))
        m.x2 = Var(bounds=(1, 7))
        # Here's a cute paraboloid:
        m.f = m.x1**2 + m.x2**2
