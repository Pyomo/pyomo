import unittest
from unittest.mock import MagicMock
from pyomo.solvers.plugins.solvers.GUROBI_RUN import gurobi_run
from unittest.mock import patch, MagicMock

try:
    from gurobipy import GRB
    gurobipy_available = True
except:
    gurobipy_available = False

class GurobiTest(unittest.TestCase):



    @unittest.skipIf(not gurobipy_available,
                     "gurobipy is not available")
    @unittest.skipIf(not hasattr(GRB, "WORK_LIMIT"),
                     "gurobi < 9.5")
    @patch("builtins.open")
    @patch("pyomo.solvers.plugins.solvers.GUROBI_RUN.read")
    def test_work_limit(self, read:MagicMock, open: MagicMock):
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
        self.assertEqual(file.write.call_args.args[0], 'termination_message: Optimization terminated because the work expended exceeded the value specified in the WorkLimit parameter.\n')


if __name__ == '__main__':
    unittest.main()
