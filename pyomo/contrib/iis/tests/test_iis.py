import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.common.tempfiles import TempfileManager

import os


def _get_infeasible_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.Binary)
    m.y = pyo.Var(within=pyo.NonNegativeReals)

    m.c1 = pyo.Constraint(expr=m.y <= 100.0 * m.x)
    m.c2 = pyo.Constraint(expr=m.y <= -100.0 * m.x)
    m.c3 = pyo.Constraint(expr=m.x >= 0.5)

    m.o = pyo.Objective(expr=-m.y)

    return m


class TestIIS(unittest.TestCase):
    @unittest.skipUnless(
        pyo.SolverFactory("cplex_persistent").available(exception_flag=False),
        "CPLEX not available",
    )
    def test_write_iis_cplex(self):
        _test_iis("cplex")

    @unittest.skipUnless(
        pyo.SolverFactory("gurobi_persistent").available(exception_flag=False),
        "Gurobi not available",
    )
    def test_write_iis_gurobi(self):
        _test_iis("gurobi")

    @unittest.skipUnless(
        pyo.SolverFactory("xpress_persistent").available(exception_flag=False),
        "Xpress not available",
    )
    def test_write_iis_xpress(self):
        _test_iis("xpress")


def _test_iis(solver_name):
    m = _get_infeasible_model()
    TempfileManager.push()
    tmp_path = TempfileManager.create_tempdir()
    file_name = os.path.join(tmp_path, f"{solver_name}_iis.ilp")
    file_name = write_iis(m, solver=solver_name, iis_file_name=str(file_name))
    _validate_ilp(file_name)
    TempfileManager.pop()


def _validate_ilp(file_name):
    lines_found = {
        "c2: 100 x + y <= 0": False,
        "c3: x >= 0.5": False,
    }
    with open(file_name, "r") as f:
        for line in f.readlines():
            for k, v in lines_found.items():
                if (not v) and k in line:
                    lines_found[k] = True

    if not all(lines_found.values()):
        raise Exception(
            f"The file {file_name} is not as expected. Missing constraints:\n"
            + "\n".join(k for k, v in lines_found.items() if not v)
        )


if __name__ == "__main__":
    unittest.main()
