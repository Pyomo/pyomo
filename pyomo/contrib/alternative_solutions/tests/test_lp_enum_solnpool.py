import pyomo.environ as pe
import pyomo.opt

import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.contrib.alternative_solutions import lp_enum
from pyomo.contrib.alternative_solutions import lp_enum_solnpool

try:
    import numpy as np

    numpy_available = True
except:
    numpy_available = False

if numpy_available:
    n = tc.get_pentagonal_pyramid_mip()
    n.x.domain = pe.Reals
    n.y.domain = pe.Reals

    sols = lp_enum_solnpool.enumerate_linear_solutions_soln_pool(n, tee=True)

    for s in sols:
        print(s)
    assert len(sols) == 6
