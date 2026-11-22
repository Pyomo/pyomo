# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from os.path import join, normpath

import pyomo.common.unittest as unittest

from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.environ import SolverFactory
from pyomo.opt import TerminationCondition

exdir = normpath(join(PYOMO_ROOT_DIR, 'examples', 'performance', 'pmedian'))

opt_glpk = SolverFactory('glpk')
glpk_available = opt_glpk.available(exception_flag=False)


@unittest.pytest.mark.solver("glpk")
class GLPKTests(unittest.TestCase):
    @unittest.skipIf(not glpk_available, "The GLPK solver is not available")
    def test_lp_time_limit_not_reported_optimal(self):
        # There is a bug where GLPK reports "optimal" when
        # a solver is actually hitting a time limit.
        # Using pmedian because it seems like a logical model for
        # testing against.
        pmedian = import_file(join(exdir, 'pmedian1.py'))
        m = pmedian.pyomo_create_model().create_instance(
            join(exdir, 'pmedian.test6.dat')
        )

        results = opt_glpk.solve(
            m, timelimit=1, options={'nopresol': None}, load_solutions=False
        )

        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxTimeLimit
        )


if __name__ == "__main__":
    unittest.main()
