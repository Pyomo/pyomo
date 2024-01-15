#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for pyomo.opt.base.solution
#

import json
import pickle
import os
from os.path import abspath, dirname, join

pyomodir = dirname(abspath(__file__)) + os.sep + ".." + os.sep + ".." + os.sep
currdir = dirname(abspath(__file__)) + os.sep

from filecmp import cmp
import pyomo.common.unittest as unittest

from pyomo.common.tempfiles import TempfileManager
import pyomo.opt

from pyomo.common.dependencies import yaml, yaml_available

old_tempdir = TempfileManager.tempdir


class Test(unittest.TestCase):
    def setUp(self):
        TempfileManager.tempdir = currdir
        self.results = pyomo.opt.SolverResults()
        self.soln = self.results.solution.add()
        self.soln.variable[1] = {"Value": 0}
        self.soln.variable[2] = {"Value": 0}
        self.soln.variable[4] = {"Value": 0}

    def tearDown(self):
        TempfileManager.clear_tempfiles()
        TempfileManager.tempdir = old_tempdir
        del self.results

    def test_write_solution1(self):
        """Write a SolverResults Object with solutions"""
        self.results.write(filename=join(currdir, "write_solution1.txt"))
        if not os.path.exists(join(currdir, "write_solution1.txt")):
            self.fail("test_write_solution - failed to write write_solution1.txt")
        _log, _out = join(currdir, "write_solution1.txt"), join(
            currdir, "test1_soln.txt"
        )
        self.assertTrue(cmp(_out, _log), msg="Files %s and %s differ" % (_out, _log))

    def test_write_solution2(self):
        """Write a SolverResults Object without solutions"""
        self.results.write(num=None, filename=join(currdir, "write_solution2.txt"))
        if not os.path.exists(join(currdir, "write_solution2.txt")):
            self.fail("test_write_solution - failed to write write_solution2.txt")
        _out, _log = join(currdir, "write_solution2.txt"), join(
            currdir, "test2_soln.txt"
        )
        self.assertTrue(cmp(_out, _log), msg="Files %s and %s differ" % (_out, _log))

    @unittest.skipIf(not yaml_available, "Cannot import 'yaml'")
    def test_read_solution1(self):
        """Read a SolverResults Object"""
        self.results = pyomo.opt.SolverResults()
        self.results.read(filename=join(currdir, "test4_sol.txt"))
        self.results.write(filename=join(currdir, "read_solution1.out"))
        if not os.path.exists(join(currdir, "read_solution1.out")):
            self.fail("test_read_solution1 - failed to write read_solution1.out")
        with open(join(currdir, "read_solution1.out"), 'r') as out, open(
            join(currdir, "test4_sol.txt"), 'r'
        ) as txt:
            self.assertStructuredAlmostEqual(
                yaml.full_load(txt), yaml.full_load(out), allow_second_superset=True
            )

    @unittest.skipIf(not yaml_available, "Cannot import 'yaml'")
    def test_pickle_solution1(self):
        """Read a SolverResults Object"""
        self.results = pyomo.opt.SolverResults()
        self.results.read(filename=join(currdir, "test4_sol.txt"))
        str = pickle.dumps(self.results)
        res = pickle.loads(str)
        self.results.write(filename=join(currdir, "read_solution1.out"))
        if not os.path.exists(join(currdir, "read_solution1.out")):
            self.fail("test_read_solution1 - failed to write read_solution1.out")
        with open(join(currdir, "read_solution1.out"), 'r') as out, open(
            join(currdir, "test4_sol.txt"), 'r'
        ) as txt:
            self.assertStructuredAlmostEqual(
                yaml.full_load(txt), yaml.full_load(out), allow_second_superset=True
            )

    def test_read_solution2(self):
        """Read a SolverResults Object"""
        self.results = pyomo.opt.SolverResults()
        self.results.read(filename=join(currdir, "test4_sol.jsn"), format='json')
        self.results.write(filename=join(currdir, "read_solution2.out"), format='json')
        if not os.path.exists(join(currdir, "read_solution2.out")):
            self.fail("test_read_solution2 - failed to write read_solution2.out")
        with open(join(currdir, "read_solution2.out"), 'r') as out, open(
            join(currdir, "test4_sol.jsn"), 'r'
        ) as txt:
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), allow_second_superset=True
            )

    def test_pickle_solution2(self):
        """Read a SolverResults Object"""
        self.results = pyomo.opt.SolverResults()
        self.results.read(filename=join(currdir, "test4_sol.jsn"), format='json')
        str = pickle.dumps(self.results)
        res = pickle.loads(str)
        self.results.write(filename=join(currdir, "read_solution2.out"), format='json')
        if not os.path.exists(join(currdir, "read_solution2.out")):
            self.fail("test_read_solution2 - failed to write read_solution2.out")
        with open(join(currdir, "read_solution2.out"), 'r') as out, open(
            join(currdir, "test4_sol.jsn"), 'r'
        ) as txt:
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), allow_second_superset=True
            )

    #
    # deleting is not supported right now
    #
    def Xtest_delete_solution(self):
        """Delete a solution from a SolverResults object"""
        self.results.solution.delete(0)
        self.results.write(filename=join(currdir, "delete_solution.txt"))
        if not os.path.exists(join(currdir, "delete_solution.txt")):
            self.fail("test_write_solution - failed to write delete_solution.txt")
        _out, _log = join(currdir, "delete_solution.txt"), join(
            currdir, "test4_soln.txt"
        )
        self.assertTrue(cmp(_out, _log), msg="Files %s and %s differ" % (_out, _log))

    def test_get_solution(self):
        """Get a solution from a SolverResults object"""
        tmp = self.results.solution[0]
        self.assertEqual(tmp, self.soln)

    def test_get_solution_attr_error(self):
        """Create an error with a solution suffix"""
        try:
            tmp = self.soln.bad
            self.fail("Expected attribute error failure for 'bad'")
        except AttributeError:
            pass

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

    def test_soln_pprint1(self):
        """Write a solution with only zero values, using the results 'write()' method"""
        self.soln.variable[1]["Value"] = 0.0
        self.soln.variable[2]["Value"] = 0.0
        self.soln.variable[4]["Value"] = 0.0
        self.results.write(filename=join(currdir, "soln_pprint.txt"))
        if not os.path.exists(join(currdir, "soln_pprint.txt")):
            self.fail("test_write_solution - failed to write soln_pprint.txt")
        _out, _log = join(currdir, "soln_pprint.txt"), join(currdir, "test3_soln.txt")
        self.assertTrue(cmp(_out, _log), msg="Files %s and %s differ" % (_out, _log))

    def test_soln_pprint2(self):
        """Write a solution with only zero values, using the Solution.pprint() method"""
        self.soln.variable[1]["Value"] = 0.0
        self.soln.variable[2]["Value"] = 0.0
        self.soln.variable[4]["Value"] = 0.0
        with open(join(currdir, 'soln_pprint2.out'), 'w') as f:
            f.write(str(self.soln))
        with open(join(currdir, "soln_pprint2.out"), 'r') as f1, open(
            join(currdir, "soln_pprint2.txt"), 'r'
        ) as f2:
            self.assertEqual(f1.read().strip(), f2.read().strip())

    def test_soln_suffix_getiter(self):
        self.soln.variable[1]["Value"] = 0.0
        self.soln.variable[2]["Value"] = 0.1
        self.soln.variable[4]["Value"] = 0.3
        self.assertEqual(self.soln.variable[4]["Value"], 0.3)
        self.assertEqual(self.soln.variable[2]["Value"], 0.1)

    def test_soln_suffix_setattr(self):
        self.soln.variable[1]["Value"] = 0.0
        self.soln.variable[4]["Value"] = 0.3
        self.soln.variable[4]["Slack"] = 0.4
        self.assertEqual(list(self.soln.variable.keys()), [1, 2, 4])
        self.assertEqual(self.soln.variable[1]["Value"], 0.0)
        self.assertEqual(self.soln.variable[4]["Value"], 0.3)
        self.assertEqual(self.soln.variable[4]["Slack"], 0.4)


if __name__ == "__main__":
    unittest.main()
