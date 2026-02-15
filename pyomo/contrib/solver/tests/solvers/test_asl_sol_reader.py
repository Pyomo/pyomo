# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import io

import pyomo.environ as pyo
from pyomo.common import unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.fileutils import this_file_dir
from pyomo.contrib.solver.solvers.asl_sol_reader import (
    ASLSolFileSolutionLoader,
    ASLSolFileData,
    asl_solve_code_to_solution_status,
    parse_asl_sol_file,
)
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.util import SolverError
from pyomo.repn.plugins.nl_writer import NLWriterInfo, ScalingFactors


class TestSolFileData(unittest.TestCase):
    def test_default_instantiation(self):
        instance = ASLSolFileData()
        self.assertEqual(instance.message, None)
        self.assertEqual(instance.objno, 0)
        self.assertEqual(instance.solve_code, None)
        self.assertEqual(instance.ampl_options, None)
        self.assertEqual(instance.primals, None)
        self.assertEqual(instance.duals, None)
        self.assertEqual(instance.var_suffixes, {})
        self.assertEqual(instance.con_suffixes, {})
        self.assertEqual(instance.obj_suffixes, {})
        self.assertEqual(instance.problem_suffixes, {})
        self.assertEqual(instance.unparsed, None)


class TestSolParserUtils(unittest.TestCase):
    def test_solve_code_to_status(self):
        sol_data = ASLSolFileData()
        sol_data.message = None

        sol_data.solve_code = None
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.noSolution, r.solution_status)
        self.assertEqual(TerminationCondition.error, r.termination_condition)
        self.assertEqual(
            "AMPL(None): solver did not generate a SOL file",
            r.extra_info.solver_message,
        )

        sol_data.primals = [1]
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.unknown, r.solution_status)
        self.assertEqual(TerminationCondition.error, r.termination_condition)
        self.assertEqual(
            "AMPL(None): solver did not generate a SOL file",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 0
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.optimal, r.solution_status)
        self.assertEqual(
            TerminationCondition.convergenceCriteriaSatisfied, r.termination_condition
        )
        self.assertEqual("", r.extra_info.solver_message)

        sol_data.solve_code = 100
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.feasible, r.solution_status)
        self.assertEqual(TerminationCondition.error, r.termination_condition)
        self.assertEqual(
            "AMPL(100:solved?): optimal solution indicated, but error likely",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 200
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.infeasible, r.solution_status)
        self.assertEqual(
            TerminationCondition.locallyInfeasible, r.termination_condition
        )
        self.assertEqual(
            "AMPL(200:infeasible): constraints cannot be satisfied",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 300
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.unknown, r.solution_status)
        self.assertEqual(TerminationCondition.unbounded, r.termination_condition)
        self.assertEqual(
            "AMPL(300:unbounded): objective can be improved without limit",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 400
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.unknown, r.solution_status)
        self.assertEqual(TerminationCondition.iterationLimit, r.termination_condition)
        self.assertEqual(
            "AMPL(400:limit): stopped by a limit that you set",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 500
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.unknown, r.solution_status)
        self.assertEqual(TerminationCondition.error, r.termination_condition)
        self.assertEqual(
            "AMPL(500:failure): stopped by an error condition in the solver",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 99
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.optimal, r.solution_status)
        self.assertEqual(
            TerminationCondition.convergenceCriteriaSatisfied, r.termination_condition
        )
        self.assertEqual("", r.extra_info.solver_message)

        sol_data.solve_code = 199
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.feasible, r.solution_status)
        self.assertEqual(TerminationCondition.error, r.termination_condition)
        self.assertEqual(
            "AMPL(199:solved?): optimal solution indicated, but error likely",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 299
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.infeasible, r.solution_status)
        self.assertEqual(
            TerminationCondition.locallyInfeasible, r.termination_condition
        )
        self.assertEqual(
            "AMPL(299:infeasible): constraints cannot be satisfied",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 399
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.unknown, r.solution_status)
        self.assertEqual(TerminationCondition.unbounded, r.termination_condition)
        self.assertEqual(
            "AMPL(399:unbounded): objective can be improved without limit",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 499
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.unknown, r.solution_status)
        self.assertEqual(TerminationCondition.iterationLimit, r.termination_condition)
        self.assertEqual(
            "AMPL(499:limit): stopped by a limit that you set",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = 599
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.unknown, r.solution_status)
        self.assertEqual(TerminationCondition.error, r.termination_condition)
        self.assertEqual(
            "AMPL(599:failure): stopped by an error condition in the solver",
            r.extra_info.solver_message,
        )

        sol_data.solve_code = -1
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.unknown, r.solution_status)
        self.assertEqual(TerminationCondition.error, r.termination_condition)
        self.assertEqual("AMPL(-1): unexpected solve code", r.extra_info.solver_message)

        sol_data.solve_code = 600
        r = Results()
        asl_solve_code_to_solution_status(sol_data, r)
        self.assertEqual(SolutionStatus.unknown, r.solution_status)
        self.assertEqual(TerminationCondition.error, r.termination_condition)
        self.assertEqual(
            "AMPL(600): unexpected solve code", r.extra_info.solver_message
        )


class TestSolParser(unittest.TestCase):
    def test_parse_minimal_sol_file(self):
        # Build a tiny .sol text stream:
        # - "Options" block with number_of_options = 2, then 4 model_object ints
        #   model_objects[1] = #cons, model_objects[3] = #vars
        # - #cons duals lines
        # - #vars primals lines
        # - "objno <i> <exit_code>"
        n_cons = 2
        n_vars = 3
        stream = io.StringIO(f"""Solver message preamble
Options
2
1
2
{n_cons}
{n_cons}
{n_vars}
{n_vars}
1.5
-2.25
10.0
20.0
30.0
objno 0 100""")
        sol_data = parse_asl_sol_file(stream)

        self.assertEqual("Solver message preamble", sol_data.message)
        self.assertEqual(0, sol_data.objno)
        self.assertEqual(100, sol_data.solve_code)
        self.assertEqual([1, 2], sol_data.ampl_options)
        self.assertEqual([10.0, 20.0, 30.0], sol_data.primals)
        self.assertEqual([1.5, -2.25], sol_data.duals)
        self.assertEqual({}, sol_data.var_suffixes)
        self.assertEqual({}, sol_data.con_suffixes)
        self.assertEqual({}, sol_data.obj_suffixes)
        self.assertEqual({}, sol_data.problem_suffixes)
        self.assertEqual(None, sol_data.unparsed)

    def test_parse_vbtol(self):
        stream = io.StringIO(f"""Solver message preamble
Options
2
1
3
2
0
3
0
1.5
objno 0 100""")
        sol_data = parse_asl_sol_file(stream)

        self.assertEqual("Solver message preamble", sol_data.message)
        self.assertEqual(0, sol_data.objno)
        self.assertEqual(100, sol_data.solve_code)
        self.assertEqual([1, 3, 1.5], sol_data.ampl_options)
        self.assertEqual([], sol_data.primals)
        self.assertEqual([], sol_data.duals)
        self.assertEqual({}, sol_data.var_suffixes)
        self.assertEqual({}, sol_data.con_suffixes)
        self.assertEqual({}, sol_data.obj_suffixes)
        self.assertEqual({}, sol_data.problem_suffixes)
        self.assertEqual(None, sol_data.unparsed)

    def test_multiline_message_and_unparsed(self):
        stream = io.StringIO("""CONOPT 3.17A: Optimal; objective 1
4 iterations; evals: nf = 2, ng = 0, nc = 2, nJ = 0, nH = 0, nHv = 0

Options
3
1
1
0
1
1
1
1
1
1
objno 0 0
suffix 0 1 8 0 0
sstatus
0 1
suffix 1 1 8 0 0
sstatus
0 3
extra data here
and here
""")
        sol_data = parse_asl_sol_file(stream)

        self.assertEqual(
            "CONOPT 3.17A: Optimal; objective 1\n"
            "4 iterations; evals: nf = 2, ng = 0, nc = 2, nJ = 0, nH = 0, nHv = 0",
            sol_data.message,
        )
        self.assertEqual(0, sol_data.objno)
        self.assertEqual(0, sol_data.solve_code)
        self.assertEqual([1, 1, 0], sol_data.ampl_options)
        self.assertEqual([1.0], sol_data.primals)
        self.assertEqual([1.0], sol_data.duals)
        self.assertEqual({'sstatus': {0: 1}}, sol_data.var_suffixes)
        self.assertEqual({'sstatus': {0: 3}}, sol_data.con_suffixes)
        self.assertEqual({}, sol_data.obj_suffixes)
        self.assertEqual({}, sol_data.problem_suffixes)
        self.assertEqual("extra data here\nand here\n", sol_data.unparsed)

    def test_suffix_table(self):
        stream = io.StringIO("""CONOPT 3.17A: Optimal; objective 1
4 iterations; evals: nf = 2, ng = 0, nc = 2, nJ = 0, nH = 0, nHv = 0

Options
3
1
1
0
1
1
1
1
1
1
objno 0 0
suffix 0 1 7 36 3
custom
1 INT An int field
2 DBL double
3 STR
0 1
suffix 1 1 8 0 0
sstatus
0 3
suffix 2 1 8 0 0
sstatus
0 2
suffix 3 1 8 0 0
sstatus
0 4

""")
        sol_data = parse_asl_sol_file(stream)

        self.assertEqual(
            "CONOPT 3.17A: Optimal; objective 1\n"
            "4 iterations; evals: nf = 2, ng = 0, nc = 2, nJ = 0, nH = 0, nHv = 0",
            sol_data.message,
        )
        self.assertEqual(0, sol_data.objno)
        self.assertEqual(0, sol_data.solve_code)
        self.assertEqual([1, 1, 0], sol_data.ampl_options)
        self.assertEqual([1.0], sol_data.primals)
        self.assertEqual([1.0], sol_data.duals)
        self.assertEqual({'custom': {0: 1}}, sol_data.var_suffixes)
        self.assertEqual({'sstatus': {0: 3}}, sol_data.con_suffixes)
        self.assertEqual({'sstatus': {0: 2}}, sol_data.obj_suffixes)
        self.assertEqual({'sstatus': 4}, sol_data.problem_suffixes)
        self.assertEqual(
            {
                (0, 'custom'): [
                    [1, 'INT', 'An int field'],
                    [2, 'DBL', 'double'],
                    [3, 'STR'],
                ]
            },
            sol_data.suffix_table,
        )
        self.assertEqual(None, sol_data.unparsed)

    def test_error_missing_options(self):
        # No line contains the substring "Options"
        bad_text = "Solver message preamble\nNo header here\n"
        stream = io.StringIO(bad_text)

        with self.assertRaisesRegex(
            SolverError, "Error reading `sol` file: no 'Options' line found."
        ):
            parse_asl_sol_file(stream)

    def test_error_malformed_options(self):
        # Contains "Options" but the required integer line is missing/blank
        bad_text = "Preamble\nOptions\n\n"
        stream = io.StringIO(bad_text)

        with self.assertRaisesRegex(ValueError, "invalid literal"):
            parse_asl_sol_file(stream)

    def test_error_objno_not_found(self):
        stream = io.StringIO(f"""Solver message preamble
Options
2
1
2
2
0
3
0
1.5
objno 0""")

        with self.assertRaisesRegex(
            SolverError,
            "Error reading `sol` file: expected 'objno'; " "received '1.5'.",
        ):
            sol_data = parse_asl_sol_file(stream)

    def test_error_objno_bad_format(self):
        stream = io.StringIO(f"""Solver message preamble
Options
2
1
2
2
0
3
0
objno 0""")

        with self.assertRaisesRegex(
            SolverError,
            "Error reading `sol` file: expected two numbers in 'objno' line; "
            "received 'objno 0'.",
        ):
            sol_data = parse_asl_sol_file(stream)


class TestSolFileSolutionLoader(unittest.TestCase):

    def test_member_list(self):
        expected_list = ['load_vars', 'get_primals', 'get_duals', 'get_reduced_costs']
        method_list = [
            method
            for method in dir(ASLSolFileSolutionLoader)
            if not method.startswith('_')
        ]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_load_vars(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])

        nl_info = NLWriterInfo(var=[m.x, m.y[1], m.y[3]])
        sol_data = ASLSolFileData()
        sol_data.primals = [3, 7, 5]
        loader = ASLSolFileSolutionLoader(sol_data, nl_info)

        loader.load_vars()
        self.assertEqual(m.x.value, 3)
        self.assertEqual(m.y[1].value, 7)
        self.assertEqual(m.y[2].value, None)
        self.assertEqual(m.y[3].value, 5)

        sol_data.primals = [13, 17, 15]
        loader.load_vars(vars_to_load=[m.y[3], m.x])
        self.assertEqual(m.x.value, 13)
        self.assertEqual(m.y[1].value, 7)
        self.assertEqual(m.y[2].value, None)
        self.assertEqual(m.y[3].value, 15)

        nl_info.scaling = ScalingFactors([1, 5, 10], [], [])
        loader.load_vars()
        self.assertEqual(m.x.value, 13)
        self.assertEqual(m.y[1].value, 3.4)
        self.assertEqual(m.y[2].value, None)
        self.assertEqual(m.y[3].value, 1.5)

        nl_info.eliminated_vars = [(m.y[2], 2 * m.y[3] + 1)]
        loader.load_vars()
        self.assertEqual(m.x.value, 13)
        self.assertEqual(m.y[1].value, 3.4)
        self.assertEqual(m.y[2].value, 4)
        self.assertEqual(m.y[3].value, 1.5)

    def test_load_vars_empty_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])

        nl_info = NLWriterInfo(
            var=[], eliminated_vars=[(m.y[3], 1.5), (m.y[2], 2 * m.y[3] + 1)]
        )
        sol_data = ASLSolFileData()
        sol_data.primals = []
        loader = ASLSolFileSolutionLoader(sol_data, nl_info)

        loader.load_vars()
        self.assertEqual(m.x.value, None)
        self.assertEqual(m.y[1].value, None)
        self.assertEqual(m.y[2].value, 4)
        self.assertEqual(m.y[3].value, 1.5)

    def test_get_primals(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])

        nl_info = NLWriterInfo(var=[m.x, m.y[1], m.y[3]])
        sol_data = ASLSolFileData()
        sol_data.primals = [3, 7, 5]
        loader = ASLSolFileSolutionLoader(sol_data, nl_info)

        self.assertEqual(
            loader.get_primals(), ComponentMap([(m.x, 3), (m.y[1], 7), (m.y[3], 5)])
        )
        self.assertEqual(m.x.value, None)
        self.assertEqual(m.y[1].value, None)
        self.assertEqual(m.y[2].value, None)
        self.assertEqual(m.y[3].value, None)

        sol_data.primals = [13, 17, 15]
        self.assertEqual(
            loader.get_primals(vars_to_load=[m.y[3], m.x]),
            ComponentMap([(m.x, 13), (m.y[3], 15)]),
        )
        self.assertEqual(m.x.value, None)
        self.assertEqual(m.y[1].value, None)
        self.assertEqual(m.y[2].value, None)
        self.assertEqual(m.y[3].value, None)

        nl_info.scaling = ScalingFactors([1, 5, 10], [], [])
        self.assertEqual(
            loader.get_primals(),
            ComponentMap([(m.x, 13), (m.y[1], 3.4), (m.y[3], 1.5)]),
        )
        self.assertEqual(m.x.value, None)
        self.assertEqual(m.y[1].value, None)
        self.assertEqual(m.y[2].value, None)
        self.assertEqual(m.y[3].value, None)

        nl_info.eliminated_vars = [(m.y[2], 2 * m.y[3] + 1)]
        self.assertEqual(
            loader.get_primals(),
            ComponentMap([(m.x, 13), (m.y[1], 3.4), (m.y[2], 4), (m.y[3], 1.5)]),
        )
        self.assertEqual(m.x.value, None)
        self.assertEqual(m.y[1].value, None)
        self.assertEqual(m.y[2].value, None)
        self.assertEqual(m.y[3].value, None)

    def test_get_primals_empty_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])

        nl_info = NLWriterInfo(
            var=[], eliminated_vars=[(m.y[3], 1.5), (m.y[2], 2 * m.y[3] + 1)]
        )
        sol_data = ASLSolFileData()
        sol_data.primals = []
        loader = ASLSolFileSolutionLoader(sol_data, nl_info)

        self.assertEqual(
            loader.get_primals(), ComponentMap([(m.y[2], 4), (m.y[3], 1.5)])
        )
        self.assertEqual(m.x.value, None)
        self.assertEqual(m.y[1].value, None)
        self.assertEqual(m.y[2].value, None)
        self.assertEqual(m.y[3].value, None)
