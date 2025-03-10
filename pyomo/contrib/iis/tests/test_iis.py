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

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
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

    @unittest.skipUnless(
        (
            pyo.SolverFactory("cplex_persistent").available(exception_flag=False)
            or pyo.SolverFactory("gurobi_persistent").available(exception_flag=False)
            or pyo.SolverFactory("xpress_persistent").available(exception_flag=False)
        ),
        "Persistent solver not available",
    )
    def test_write_iis_any_solver(self):
        _test_iis(None)

    @unittest.skipIf(
        pyo.SolverFactory("cplex_persistent").available(exception_flag=False),
        "CPLEX available",
    )
    def test_exception_cplex_not_available(self):
        self._assert_raises_unavailable_solver("cplex")

    @unittest.skipIf(
        pyo.SolverFactory("gurobi_persistent").available(exception_flag=False),
        "Gurobi available",
    )
    def test_exception_gurobi_not_available(self):
        self._assert_raises_unavailable_solver("gurobi")

    @unittest.skipIf(
        pyo.SolverFactory("xpress_persistent").available(exception_flag=False),
        "Xpress available",
    )
    def test_exception_xpress_not_available(self):
        self._assert_raises_unavailable_solver("xpress")

    @unittest.skipIf(
        (
            pyo.SolverFactory("cplex_persistent").available(exception_flag=False)
            or pyo.SolverFactory("gurobi_persistent").available(exception_flag=False)
            or pyo.SolverFactory("xpress_persistent").available(exception_flag=False)
        ),
        "Persistent solver available",
    )
    def test_exception_iis_no_solver_available(self):
        with self.assertRaises(
            RuntimeError,
            msg=f"Could not find a solver to use, supported solvers are {_supported_solvers}",
        ):
            _test_iis(None)

    def _assert_raises_unavailable_solver(self, solver_name):
        with self.assertRaises(
            RuntimeError,
            msg=f"The Pyomo persistent interface to {solver_name} could not be found.",
        ):
            _test_iis(solver_name)


def _test_iis(solver_name):
    m = _get_infeasible_model()
    TempfileManager.push()
    tmp_path = TempfileManager.create_tempdir()
    file_name = os.path.join(tmp_path, f"{solver_name}_iis.ilp")
    file_name = write_iis(m, solver=solver_name, iis_file_name=str(file_name))
    _validate_ilp(file_name)
    TempfileManager.pop()


def _validate_ilp(file_name):
    lines_found = {"c2: 100 x + y <= 0": False, "c3: x >= 0.5": False}
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
