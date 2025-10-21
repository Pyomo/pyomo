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

import pyomo.environ as pyo
from pyomo.common import unittest
from pyomo.common.errors import PyomoException
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.contrib.solver.solvers.sol_reader import (
    SolSolutionLoader,
    SolFileData,
    parse_sol_file,
)
from pyomo.contrib.solver.common.results import Results

currdir = this_file_dir()


class TestSolFileData(unittest.TestCase):
    def test_default_instantiation(self):
        instance = SolFileData()
        self.assertIsInstance(instance.primals, list)
        self.assertIsInstance(instance.duals, list)
        self.assertIsInstance(instance.var_suffixes, dict)
        self.assertIsInstance(instance.con_suffixes, dict)
        self.assertIsInstance(instance.obj_suffixes, dict)
        self.assertIsInstance(instance.problem_suffixes, dict)
        self.assertIsInstance(instance.other, list)


class TestSolParser(unittest.TestCase):
    def setUp(self):
        TempfileManager.push()

    def tearDown(self):
        TempfileManager.pop(remove=True)

    class _FakeNLInfo:
        def __init__(
            self,
            variables,
            constraints,
            objectives=None,
            scaling=None,
            eliminated_vars=None,
        ):
            self.variables = variables
            self.constraints = constraints
            self.objectives = objectives or []
            self.scaling = scaling
            self.eliminated_vars = eliminated_vars or []

    class _FakeSolData:
        def __init__(self, primals=None, duals=None):
            self.primals = primals or []
            self.duals = duals or []
            self.var_suffixes = {}
            self.con_suffixes = {}
            self.obj_suffixes = {}
            self.problem_suffixes = {}
            self.other = []

    def test_get_duals_no_objective_returns_zeros(self):
        # model with 2 cons, no objective
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.y = pyo.Var(initialize=2.0)
        m.c1 = pyo.Constraint(expr=m.x + m.y >= 0)
        m.c2 = pyo.Constraint(expr=m.x - m.y <= 3)

        nl_info = self._FakeNLInfo(
            variables=[m.x, m.y], constraints=[m.c1, m.c2], objectives=[], scaling=None
        )
        # solver returned some (non-zero) duals, but we should zero them out
        sol_data = self._FakeSolData(duals=[123.0, -7.5])

        loader = SolSolutionLoader(sol_data, nl_info)
        duals = loader.get_duals()
        self.assertEqual(duals[m.c1], 0.0)
        self.assertEqual(duals[m.c2], 0.0)

    def test_parse_sol_file(self):
        # Build a tiny .sol text stream:
        # - "Options" block with number_of_options = 0, then 4 model_object ints
        #   model_objects[1] = #cons, model_objects[3] = #vars
        # - #cons duals lines
        # - #vars primals lines
        # - "objno <i> <exit_code>"
        n_cons = 2
        n_vars = 3
        sol_content = (
            "Solver message preamble\n"
            "Options\n"
            "0\n"
            f"0\n{n_cons}\n0\n{n_vars}\n"  # model_objects (4 ints)
            "1.5\n-2.25\n"  # duals (2 lines)
            "10.0\n20.0\n30.0\n"  # primals (3 lines)
            "objno 0 0\n"  # exit code line
        )
        stream = io.StringIO(sol_content)

        # Minimal NL info matching sizes
        m = pyo.ConcreteModel()
        m.v = pyo.Var(range(n_vars))
        m.c = pyo.Constraint(range(n_cons), rule=lambda m, i: m.v[0] >= -100)
        nl_info = self._FakeNLInfo(
            variables=[m.v[i] for i in range(n_vars)],
            constraints=[m.c[i] for i in range(n_cons)],
        )

        res = Results()
        res_out, sol_data = parse_sol_file(stream, nl_info, res)

        # Check counts populated
        self.assertEqual(len(sol_data.duals), n_cons)
        self.assertEqual(len(sol_data.primals), n_vars)
        # Exit code 0..99 -> optimal + convergenceCriteriaSatisfied
        self.assertEqual(res_out.solution_status.name, "optimal")
        self.assertEqual(
            res_out.termination_condition.name, "convergenceCriteriaSatisfied"
        )

        # Values preserved
        self.assertAlmostEqual(sol_data.duals[0], 1.5)
        self.assertAlmostEqual(sol_data.duals[1], -2.25)
        self.assertEqual(sol_data.primals, [10.0, 20.0, 30.0])

    def test_parse_sol_file_missing_options_raises(self):
        # No line contains the substring "Options"
        bad_text = "Solver message preamble\nNo header here\n"
        stream = io.StringIO(bad_text)

        nl_info = self._FakeNLInfo(variables=[], constraints=[])

        with self.assertRaises(PyomoException):
            parse_sol_file(stream, nl_info, Results())

    def test_parse_sol_file_malformed_options_raises(self):
        # Contains "Options" but the required integer line is missing/blank
        bad_text = "Preamble\nOptions\n\n"
        stream = io.StringIO(bad_text)

        nl_info = self._FakeNLInfo(variables=[], constraints=[])

        with self.assertRaises(ValueError):
            parse_sol_file(stream, nl_info, Results())
