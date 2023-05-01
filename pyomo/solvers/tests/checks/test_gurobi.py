import pyomo.common.unittest as unittest
from unittest.mock import patch, MagicMock

try:
    from pyomo.solvers.plugins.solvers.GUROBI_RUN import gurobi_run
    from gurobipy import GRB

    gurobipy_available = True
    has_worklimit = hasattr(GRB, "WORK_LIMIT")
except:
    gurobipy_available = False
    has_worklimit = False


@unittest.skipIf(not gurobipy_available, "gurobipy is not available")
class GurobiTest(unittest.TestCase):
    @unittest.skipIf(not has_worklimit, "gurobi < 9.5")
    @patch("builtins.open")
    @patch("pyomo.solvers.plugins.solvers.GUROBI_RUN.read")
    def test_work_limit(self, read: MagicMock, open: MagicMock):
        file = MagicMock()
        open.return_value = file
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
        gurobi_run(None, None, None, None, {}, [])
        self.assertTrue("WorkLimit" in file.write.call_args[0][0])


if __name__ == '__main__':
    unittest.main()
