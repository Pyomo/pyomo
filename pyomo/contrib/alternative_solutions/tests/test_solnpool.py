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

import pyomo.environ as pe
import pyomo.common.unittest as unittest

from pyomo.contrib.alternative_solutions.solnpool \
    import gurobi_generate_solutions
from pyomo.contrib.alternative_solutions.tests.test_cases \
    import get_aos_test_knapsack, get_triangle_ip

class TestSolnPoolUnit(unittest.TestCase):
    def test_(self):
        pass

if __name__ == '__main__':
    unittest.main()