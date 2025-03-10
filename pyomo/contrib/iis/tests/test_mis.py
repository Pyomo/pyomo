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
import pyomo.contrib.iis.mis as mis
from pyomo.contrib.iis.mis import _get_constraint
from pyomo.common.tempfiles import TempfileManager

import logging
import os


def _get_infeasible_model():
    m = pyo.ConcreteModel("trivial4test")
    m.x = pyo.Var(within=pyo.Binary)
    m.y = pyo.Var(within=pyo.NonNegativeReals)

    m.c1 = pyo.Constraint(expr=m.y <= 100.0 * m.x)
    m.c2 = pyo.Constraint(expr=m.y <= -100.0 * m.x)
    m.c3 = pyo.Constraint(expr=m.x >= 0.5)

    m.o = pyo.Objective(expr=-m.y)

    return m


def _get_feasible_model():
    m = pyo.ConcreteModel("Trivial Feasible Quad")
    m.x = pyo.Var([1, 2], bounds=(0, 1))
    m.y = pyo.Var(bounds=(0, 1))
    m.c = pyo.Constraint(expr=m.x[1] * m.x[2] >= -1)
    m.d = pyo.Constraint(expr=m.x[1] + m.y >= 1)

    return m


class TestMIS(unittest.TestCase):
    @unittest.skipUnless(
        pyo.SolverFactory("ipopt").available(exception_flag=False),
        "ipopt not available",
    )
    def test_write_mis_ipopt(self):
        _test_mis("ipopt")

    def test__get_constraint_errors(self):
        # A not-completely-cynical way to get the coverage up.
        m = _get_infeasible_model()  # not modified
        fct = _get_constraint

        m.foo_slack_plus_ = pyo.Var()
        self.assertRaises(RuntimeError, fct, m, m.foo_slack_plus_)
        m.foo_slack_minus_ = pyo.Var()
        self.assertRaises(RuntimeError, fct, m, m.foo_slack_minus_)
        m.foo_bar = pyo.Var()
        self.assertRaises(RuntimeError, fct, m, m.foo_bar)

    def test_feasible_model(self):
        m = _get_feasible_model()
        opt = pyo.SolverFactory("ipopt")
        self.assertRaises(Exception, mis.compute_infeasibility_explanation, m, opt)


def _check_output(file_name):
    # pretty simple check for now
    with open(file_name, "r+") as file1:
        lines = file1.readlines()
    trigger = "Constraints / bounds in MIS:"
    nugget = "lb of var y"
    live = False  # (long i)
    found_nugget = False
    for line in lines:
        if trigger in line:
            live = True
        if live:
            if nugget in line:
                found_nugget = True
    if not found_nugget:
        raise RuntimeError(f"Did not find '{nugget}' after '{trigger}' in output")
    else:
        pass


def _test_mis(solver_name):
    m = _get_infeasible_model()
    opt = pyo.SolverFactory(solver_name)

    # This test seems to fail on Windows as it unlinks the tempfile, so live with it
    #    On a Windows machine, we will not use a temp dir and just try to delete the log file
    if os.name == "nt":
        file_name = f"_test_mis_{solver_name}.log"
        logger = logging.getLogger(f"test_mis_{solver_name}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(file_name)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        mis.compute_infeasibility_explanation(m, opt, logger=logger)
        _check_output(file_name)
        # os.remove(file_name) cannot remove it on Windows. Still in use.

    else:  # not windows
        with TempfileManager.new_context() as tmpmgr:
            tmp_path = tmpmgr.mkdtemp()
            file_name = os.path.join(tmp_path, f"_test_mis_{solver_name}.log")
            logger = logging.getLogger(f"test_mis_{solver_name}")
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler(file_name)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)

            mis.compute_infeasibility_explanation(m, opt, logger=logger)
            _check_output(file_name)


if __name__ == "__main__":
    unittest.main()
