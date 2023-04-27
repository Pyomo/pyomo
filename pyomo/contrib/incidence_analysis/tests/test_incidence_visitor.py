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

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.contrib.incidence_analysis.interface import get_incident_variables


class TestUninitialized(unittest.TestCase):
    def test_product_one_fixed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2])
        m.x[1].fix()
        variables = get_incident_variables(m.x[1]*m.x[2])
        self.assertEqual(len(variables), 1)


if __name__ == "__main__":
    unittest.main()
