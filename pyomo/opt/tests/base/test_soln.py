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
#
# Unit Tests for pyomo.opt.results
#
import copy
import json
import pickle
import os

from io import StringIO
from os.path import join
from filecmp import cmp

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager

import pyomo.opt.results.container as container
from pyomo.opt import SolverResults

currdir = this_file_dir()

ref_yaml_soln = """\
# ==========================================================
# = Solver Results                                         =
# ==========================================================

# ----------------------------------------------------------
#   Problem Information
# ----------------------------------------------------------
Problem: 
- Lower bound: -Infinity
  Upper bound: Infinity
  Number of objectives: 1
  Number of constraints: 24
  Number of variables: 32
  Sense: unknown

# ----------------------------------------------------------
#   Solver Information
# ----------------------------------------------------------
Solver: 
- Status: ok
  Message: PICO Solver\\x3a final f = 88.200000
  Termination condition: optimal
  Id: 0

# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution: 
- number of solutions: 1
  number of solutions displayed: 1
- Status: optimal
  Variable: 
    v4:
      Value: 46.6666666667
    v11:
      Value: 933.333333333
    v12:
      Value: 10000
    v13:
      Value: 10000
    v14:
      Value: 10000
    v15:
      Value: 10000
    v17:
      Value: 100
    v19:
      Value: 100
    v21:
      Value: 100
    v23:
      Value: 100
    v24:
      Value: 46.6666666667
    v25:
      Value: 53.3333333333
    v27:
      Value: 100
    v29:
      Value: 100
    v31:
      Value: 100
  Constraint: 
    c2: 
      Dual: 0.126
  Status description: OPTIMAL SOLUTION FOUND!
"""

ref_json_soln = """\
{
    "Problem": [
        {
            "Lower bound": -Infinity, 
            "Number of constraints": 24, 
            "Number of objectives": 1, 
            "Number of variables": 32, 
            "Sense": "unknown", 
            "Upper bound": Infinity
        }
    ], 
    "Solution": [
        {
            "number of solutions": 1, 
            "number of solutions displayed": 1
        }, 
        {
            "Constraint": {
                "c2": {
                    "Dual": 0.12599999999999997
                }
            }, 
            "Message": "PICO Solver\\\\x3a final f = 88.200000", 
            "Status": "optimal", 
            "Status description": "OPTIMAL SOLUTION FOUND!", 
            "Variable": {
                "v11": {
                    "Value": 933.3333333333336
                }, 
                "v12": {
                    "Value": 10000.0
                }, 
                "v13": {
                    "Value": 10000.0
                }, 
                "v14": {
                    "Value": 10000.0
                }, 
                "v15": {
                    "Value": 10000.0
                }, 
                "v17": {
                    "Value": 100.0
                }, 
                "v19": {
                    "Value": 100.0
                }, 
                "v21": {
                    "Value": 100.0
                }, 
                "v23": {
                    "Value": 100.0
                }, 
                "v24": {
                    "Value": 46.666666666666664
                }, 
                "v25": {
                    "Value": 53.333333333333336
                }, 
                "v27": {
                    "Value": 100.0
                }, 
                "v29": {
                    "Value": 100.0
                }, 
                "v31": {
                    "Value": 100.0
                }, 
                "v4": {
                    "Value": 46.666666666666664
                }
            }
        }
    ], 
    "Solver": [
        {
            "Id": 0, 
            "Message": "PICO Solver\\\\x3a final f = 88.200000", 
            "Status": "ok", 
            "Termination condition": "optimal"
        }
    ]
}
"""


class Test(unittest.TestCase):

    def test_write_solution(self):
        results = SolverResults()
        soln = results.solution.add()
        soln.variable[1] = {"Value": 0}
        soln.variable[2] = {"Value": 0}
        soln.variable[4] = {"Value": 0}

        OUT = StringIO()
        results.write(ostream=OUT)
        self.assertEqual(
            """
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution: 
- number of solutions: 1
  number of solutions displayed: 1
- Status: unknown
  Objective: No values
  Variable: No nonzero values
  Constraint: No values
            """.strip(),
            OUT.getvalue().strip(),
        )

    def test_write_empty_solution(self):
        results = SolverResults()
        soln = results.solution.add()
        soln.variable[1] = {"Value": 0}
        soln.variable[2] = {"Value": 0}
        soln.variable[4] = {"Value": 0}

        OUT = StringIO()
        results.write(num=None, ostream=OUT)
        self.assertEqual(
            """
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution: 
- number of solutions: 1
  number of solutions displayed: 1
- Status: unknown
  Objective: No values
  Variable: No nonzero values
  Constraint: No values
            """.strip(),
            OUT.getvalue().strip(),
        )

    @unittest.skipIf(not yaml_available, "Test requires 'yaml'")
    def test_read_write_yaml_solution(self):
        """Read a SolverResults Object"""
        IN = StringIO(ref_yaml_soln)
        results = SolverResults()
        results.read(istream=IN, format='yaml')
        OUT = StringIO()
        results.write(ostream=OUT, format='yaml')

        self.assertStructuredAlmostEqual(
            yaml.full_load(StringIO(ref_yaml_soln)),
            yaml.full_load(StringIO(OUT.getvalue())),
            allow_second_superset=True,
        )

    @unittest.skipIf(not yaml_available, "Test requires 'yaml'")
    def test_pickle_yaml_solution(self):
        IN = StringIO(ref_yaml_soln)
        results = SolverResults()
        results.read(istream=IN, format='yaml')

        new_results = pickle.loads(pickle.dumps(results))
        OUT = StringIO()
        new_results.write(ostream=OUT, format='yaml')

        self.assertStructuredAlmostEqual(
            yaml.full_load(StringIO(ref_yaml_soln)),
            yaml.full_load(StringIO(OUT.getvalue())),
            allow_second_superset=True,
        )

    @unittest.skipIf(not yaml_available, "Test requires 'yaml'")
    def test_write_read_yaml_file(self):
        IN = StringIO(ref_yaml_soln)
        results = SolverResults()
        results.read(istream=IN, format='yaml')

        with TempfileManager.new_context() as temp:
            fname = temp.create_tempfile('yaml')
            results.write(filename=fname)
            new_results = SolverResults()
            new_results.read(filename=fname)

        self.assertStructuredAlmostEqual(
            results, new_results, allow_second_superset=True
        )

    def test_read_write_json_solution(self):
        IN = StringIO(ref_json_soln)
        results = SolverResults()
        results.read(istream=IN, format='json')
        OUT = StringIO()
        results.write(ostream=OUT, format='json')

        self.assertStructuredAlmostEqual(
            json.loads(ref_json_soln),
            json.loads(OUT.getvalue()),
            allow_second_superset=True,
        )

    def test_pickle_json_solution(self):
        IN = StringIO(ref_json_soln)
        results = SolverResults()
        results.read(istream=IN, format='json')

        new_results = pickle.loads(pickle.dumps(results))
        OUT = StringIO()
        new_results.write(ostream=OUT, format='json')

        self.assertStructuredAlmostEqual(
            json.loads(ref_json_soln),
            json.loads(OUT.getvalue()),
            allow_second_superset=True,
        )

    def test_write_read_json_file(self):
        IN = StringIO(ref_json_soln)
        results = SolverResults()
        results.read(istream=IN, format='json')

        with TempfileManager.new_context() as temp:
            fname = temp.create_tempfile('.jsn')
            results.write(filename=fname)
            new_results = SolverResults()
            new_results.read(filename=fname)

        self.assertStructuredAlmostEqual(
            results, new_results, allow_second_superset=True
        )

    def test_deepcopy_solution(self):
        IN = StringIO(ref_json_soln)
        result = SolverResults()
        result.read(istream=IN, format='json')
        result_copy = copy.deepcopy(result)
        self.assertEqual(list(result.keys()), list(result_copy.keys()))
        self.assertEqual(str(result), str(result_copy))
        self.assertEqual(result, result)
        self.assertEqual(result, result_copy)

    #
    # deleting is not supported right now
    #
    def test_delete_solution(self):
        results = SolverResults()
        soln = results.solution.add()
        soln.variable[1] = {"Value": 0}
        soln.variable[2] = {"Value": 0}
        soln.variable[4] = {"Value": 0}

        results.solution.delete(0)

        OUT = StringIO()
        results.write(ostream=OUT)
        self.assertEqual(
            """
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution: 
- number of solutions: 0
  number of solutions displayed: 0
            """.strip(),
            OUT.getvalue().strip(),
        )

    def test_get_solution(self):
        """Get a solution from a SolverResults object"""
        results = SolverResults()
        soln = results.solution.add()
        self.assertIs(results.solution[0], soln)

    def test_get_solution_attr_error(self):
        """Create an error with a solution suffix"""
        results = SolverResults()
        soln = results.solution.add()
        with self.assertRaisesRegex(
            AttributeError,
            r"Unknown attribute `bad' for object with type <.*\.Solution'>",
        ):
            tmp = soln.bad

    #
    # This is currently allowed, although soln.variable = True is equivalent to
    #   soln.variable.value = True
    #
    def Xtest_set_solution_attr_error(self):
        """Create an error with a solution suffix"""
        try:
            self.soln.variable = True
            self.fail("Expected attribute error failure for 'variable'")
        except AttributeError:
            pass

    def test_soln_write_zeros(self):
        results = SolverResults()
        soln = results.solution.add()
        soln.variable[1] = {"Value": 0}
        soln.variable[2] = {"Value": 0}
        soln.variable[4] = {"Value": 0}

        soln.variable[1]["Value"] = 0.0
        soln.variable[2]["Value"] = 0.0
        soln.variable[4]["Value"] = 0.0

        OUT = StringIO()
        results.write(ostream=OUT)
        self.assertEqual(
            """
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution: 
- number of solutions: 1
  number of solutions displayed: 1
- Status: unknown
  Objective: No values
  Variable: No nonzero values
  Constraint: No values
            """.strip(),
            OUT.getvalue().strip(),
        )

    def test_soln_write_nonzeros(self):
        results = SolverResults()
        soln = results.solution.add()
        soln.variable[1] = {"Value": 0}
        soln.variable[2] = {"Value": 0}
        soln.variable[4] = {"Value": 0}

        soln.variable[1]["Value"] = 1.0
        soln.variable[2]["Value"] = 3.0
        soln.variable[4]["Value"] = 5.0

        OUT = StringIO()
        results.write(ostream=OUT)
        self.assertEqual(
            """
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution: 
- number of solutions: 1
  number of solutions displayed: 1
- Status: unknown
  Objective: No values
  Variable:
    1:
      Value: 1
    2:
      Value: 3
    4:
      Value: 5
  Constraint: No values
            """.strip(),
            OUT.getvalue().strip(),
        )

    def test_soln_suffix_getattr(self):
        results = SolverResults()
        soln = results.solution.add()
        soln.variable[1] = {"Value": 0}
        soln.variable[2] = {"Value": 0}
        soln.variable[4] = {"Value": 0}

        soln.variable[1]["Value"] = 0.0
        soln.variable[2]["Value"] = 0.1
        soln.variable[4]["Value"] = 0.3
        self.assertEqual(soln.variable[4]["Value"], 0.3)
        self.assertEqual(soln.variable[2]["Value"], 0.1)

    def test_soln_suffix_setattr_getattr(self):
        results = SolverResults()
        soln = results.solution.add()
        soln.variable[1] = {"Value": 0}
        soln.variable[2] = {"Value": 0}
        soln.variable[4] = {"Value": 0}

        soln.variable[1]["Value"] = 0.0
        soln.variable[4]["Value"] = 0.3
        soln.variable[4]["Slack"] = 0.4
        self.assertEqual(list(soln.variable.keys()), [1, 2, 4])
        self.assertEqual(soln.variable[1]["Value"], 0.0)
        self.assertEqual(soln.variable[4]["Value"], 0.3)
        self.assertEqual(soln.variable[4]["Slack"], 0.4)

    def test_soln_to_str(self):
        results = SolverResults()
        soln = results.solution.add()
        soln.variable[1] = {"Value": 0}
        soln.variable[2] = {"Value": 0}
        soln.variable[4] = {"Value": 0}

        self.assertEqual(
            """
Solution: 
- number of solutions: 1
  number of solutions displayed: 1
- Status: unknown
  Objective: No values
  Variable: No nonzero values
  Constraint: No values
            """.strip(),
            str(results).strip(),
        )

        soln.variable[1]["Value"] = 1.0
        soln.variable[2]["Value"] = 3.0
        soln.variable[4]["Value"] = 5.0

        self.assertEqual(
            """
Solution: 
- number of solutions: 1
  number of solutions displayed: 1
- Status: unknown
  Objective: No values
  Variable:
    1:
      Value: 1
    2:
      Value: 3
    4:
      Value: 5
  Constraint: No values
            """.strip(),
            str(results).strip(),
        )


class TestContainer(unittest.TestCase):
    def test_declare_and_str(self):
        class LocalContainer(container.MapContainer):
            def __init__(self, **kwds):
                super().__init__(**kwds)
                self.declare('a')
                self.declare('b', value=2)
                self.declare('c', value=3)

        d = container.MapContainer()
        d.declare('f')
        d.declare('g')
        d.declare('h')
        d.declare('i', value=container.ListContainer(container.UndefinedData))
        d.declare('j', value=container.ListContainer(LocalContainer), active=False)
        self.assertEqual(list(d.keys()), ['F', 'G', 'H', 'I', 'J'])

        self.assertEqual(
            """
I: 
""",
            str(d),
        )

        # Assigning to a field activates it
        d.f = 1
        self.assertEqual(
            """
F: 1
I: 
""",
            str(d),
        )
        self.assertEqual(d.f, 1)

        # Adding to a list also activates it, even if it was declared
        # with active=False
        self.assertTrue(d.i._active)
        self.assertFalse(d.j._active)
        d.j.add()
        self.assertTrue(d.i._active)
        self.assertTrue(d.j._active)
        self.assertEqual(
            """
F: 1
I: 
J: 
- B: 2
  C: 3
""",
            str(d),
        )

        self.assertEqual(
            """
- B: 2
  C: 3
""",
            str(d.j),
        )

    def test_pickle(self):
        d = container.MapContainer()
        d.declare('a', value=container.ignore)
        d.declare('b', value=container.undefined)
        d.declare('c', value=42)
        e = pickle.loads(pickle.dumps(d))
        self.assertEqual(list(d.keys()), list(e.keys()))
        self.assertIs(d.a, e.a)
        self.assertIs(d.b, e.b)
        self.assertEqual(d.c, e.c)

    def test_eq(self):
        d = container.MapContainer()
        d.declare('x', value=1)
        d.declare('y', value='a')
        d.declare('z', value=container.ListContainer(container.MapContainer))
        self.assertFalse(d == container.MapContainer())
        self.assertFalse(d == "Something else")

        e = container.ListContainer(container.UndefinedData)
        self.assertFalse(d == e)

        dd = container.MapContainer()
        dd.declare('x', value=1)
        dd.declare('y', value=1)
        dd.declare('z', value=container.ListContainer(container.UndefinedData))
        self.assertFalse(d == dd)
        d.y = 'b'
        dd.y = 'b'
        dd.declare('z', value=container.ListContainer(container.MapContainer))
        self.assertTrue(d == dd)

        d.z.add()
        d.z[0].declare('a', value=0)
        self.assertFalse(d == dd)
        dd.z.add()
        dd.z[0].declare('a', value=0)
        self.assertTrue(d == dd)


if __name__ == "__main__":
    unittest.main()
