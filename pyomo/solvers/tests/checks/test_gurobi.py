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

import io
import pyomo.common.unittest as unittest
from unittest.mock import patch, MagicMock

try:
    from pyomo.solvers.plugins.solvers.GUROBI_RUN import gurobi_run, write_result
    from gurobipy import GRB

    gurobipy_available = True
    has_worklimit = hasattr(GRB, "WORK_LIMIT")
except:
    gurobipy_available = False
    has_worklimit = False


@unittest.skipIf(not gurobipy_available, "gurobipy is not available")
class GurobiTest(unittest.TestCase):
    @unittest.skipIf(not has_worklimit, "gurobi < 9.5")
    @patch("pyomo.solvers.plugins.solvers.GUROBI_RUN.read")
    def test_work_limit(self, read: MagicMock):
        model = MagicMock()
        read.return_value = model

        def getAttr(attr):
            if attr == GRB.Attr.Status:
                return GRB.WORK_LIMIT
            elif attr == GRB.Attr.ModelSense:
                return 1
            elif attr == GRB.Attr.ModelName:
                return ""
            elif attr.startswith("Num"):
                return 1
            elif attr == GRB.Attr.SolCount:
                return 0

            return None

        model.getAttr = getAttr
        result = gurobi_run(None, None, None, {}, [])
        self.assertIn("WorkLimit", result['solver']['message'])


if __name__ == '__main__':
    unittest.main()
