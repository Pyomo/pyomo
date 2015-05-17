#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for pyomo.opt.base.solution
#

import pickle
import os
from os.path import abspath, dirname
pyomodir = dirname(abspath(__file__))+os.sep+".."+os.sep+".."+os.sep
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.misc
import pyutilib.services

import pyomo.opt

from six import iterkeys

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

old_tempdir = pyutilib.services.TempfileManager.tempdir

class Test(unittest.TestCase):

    def setUp(self):
        pyutilib.services.TempfileManager.tempdir = currdir
        self.results = pyomo.opt.SolverResults()
        self.soln = self.results.solution.add()
        self.soln.variable[1]={"Value" : 0}
        self.soln.variable[2]={"Value" : 0}
        self.soln.variable[4]={"Value" : 0}

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir
        del self.results

    def test_write_solution1(self):
        """ Write a SolverResults Object with solutions """
        self.results.write(filename=currdir+"write_solution1.txt")
        if not os.path.exists(currdir+"write_solution1.txt"):
            self.fail("test_write_solution - failed to write write_solution1.txt")
        self.assertFileEqualsBaseline(currdir+"write_solution1.txt", currdir+"test1_soln.txt")

    def test_write_solution2(self):
        """ Write a SolverResults Object without solutions """
        self.results.write(num=None,filename=currdir+"write_solution2.txt")
        if not os.path.exists(currdir+"write_solution2.txt"):
            self.fail("test_write_solution - failed to write write_solution2.txt")
        self.assertFileEqualsBaseline(currdir+"write_solution2.txt", currdir+"test2_soln.txt")

    @unittest.skipIf(not yaml_available, "Cannot import 'yaml'")
    def test_read_solution1(self):
        """ Read a SolverResults Object"""
        self.results = pyomo.opt.SolverResults()
        self.results.read(filename=currdir+"test4_sol.txt")
        self.results.write(filename=currdir+"read_solution1.out")
        if not os.path.exists(currdir+"read_solution1.out"):
            self.fail("test_read_solution1 - failed to write read_solution1.out")
        self.assertMatchesYamlBaseline(currdir+"read_solution1.out", currdir+"test4_sol.txt")

    @unittest.skipIf(not yaml_available, "Cannot import 'yaml'")
    def test_pickle_solution1(self):
        """ Read a SolverResults Object"""
        self.results = pyomo.opt.SolverResults()
        self.results.read(filename=currdir+"test4_sol.txt")
        str = pickle.dumps(self.results)
        res = pickle.loads(str)
        self.results.write(filename=currdir+"read_solution1.out")
        if not os.path.exists(currdir+"read_solution1.out"):
            self.fail("test_read_solution1 - failed to write read_solution1.out")
        self.assertMatchesYamlBaseline(currdir+"read_solution1.out", currdir+"test4_sol.txt")

    def test_read_solution2(self):
        """ Read a SolverResults Object"""
        self.results = pyomo.opt.SolverResults()
        self.results.read(filename=currdir+"test4_sol.jsn", format='json')
        self.results.write(filename=currdir+"read_solution2.out", format='json')
        if not os.path.exists(currdir+"read_solution2.out"):
            self.fail("test_read_solution2 - failed to write read_solution2.out")
        self.assertMatchesJsonBaseline(currdir+"read_solution2.out", currdir+"test4_sol.jsn")

    def test_pickle_solution2(self):
        """ Read a SolverResults Object"""
        self.results = pyomo.opt.SolverResults()
        self.results.read(filename=currdir+"test4_sol.jsn", format='json')
        str = pickle.dumps(self.results)
        res = pickle.loads(str)
        self.results.write(filename=currdir+"read_solution2.out", format='json')
        if not os.path.exists(currdir+"read_solution2.out"):
            self.fail("test_read_solution2 - failed to write read_solution2.out")
        self.assertMatchesJsonBaseline(currdir+"read_solution2.out", currdir+"test4_sol.jsn")

    #
    # deleting is not supported right now
    #
    def Xtest_delete_solution(self):
        """ Delete a solution from a SolverResults object """
        self.results.solution.delete(0)
        self.results.write(filename=currdir+"delete_solution.txt")
        if not os.path.exists(currdir+"delete_solution.txt"):
            self.fail("test_write_solution - failed to write delete_solution.txt")
        self.assertFileEqualsBaseline(currdir+"delete_solution.txt", currdir+"test4_soln.txt")

    def test_get_solution(self):
        """ Get a solution from a SolverResults object """
        tmp = self.results.solution[0]
        self.assertEqual(tmp,self.soln)

    def test_get_solution_attr_error(self):
        """ Create an error with a solution suffix """
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
        """ Create an error with a solution suffix """
        try:
            self.soln.variable = True
            self.fail("Expected attribute error failure for 'variable'")
        except AttributeError:
            pass

    def test_soln_pprint1(self):
        """ Write a solution with only zero values, using the results 'write()' method """
        self.soln.variable[1]["Value"]=0.0
        self.soln.variable[2]["Value"]=0.0
        self.soln.variable[4]["Value"]=0.0
        self.results.write(filename=currdir+"soln_pprint.txt")
        if not os.path.exists(currdir+"soln_pprint.txt"):
            self.fail("test_write_solution - failed to write soln_pprint.txt")
        self.assertFileEqualsBaseline(currdir+"soln_pprint.txt", currdir+"test3_soln.txt")

    def test_soln_pprint2(self):
        """ Write a solution with only zero values, using the Solution.pprint() method """
        self.soln.variable[1]["Value"]=0.0
        self.soln.variable[2]["Value"]=0.0
        self.soln.variable[4]["Value"]=0.0
        pyutilib.misc.setup_redirect(currdir+"soln_pprint2.out")
        print(self.soln)
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(currdir+"soln_pprint2.out", currdir+"soln_pprint2.txt")

    def test_soln_suffix_getiter(self):
        self.soln.variable[1]["Value"]=0.0
        self.soln.variable[2]["Value"]=0.1
        self.soln.variable[4]["Value"]=0.3
        self.assertEqual(self.soln.variable[4]["Value"],0.3)
        self.assertEqual(self.soln.variable[2]["Value"],0.1)

    def test_soln_suffix_setattr(self):
        self.soln.variable[1]["Value"] = 0.0
        self.soln.variable[4]["Value"] =0.3
        self.soln.variable[4]["Slack"] = 0.4
        self.assertEqual(list(iterkeys(self.soln.variable)),[1,2,4])
        self.assertEqual(self.soln.variable[1]["Value"],0.0)
        self.assertEqual(self.soln.variable[4]["Value"],0.3)
        self.assertEqual(self.soln.variable[4]["Slack"],0.4)

if __name__ == "__main__":
    import pyutilib.misc
    #sys.settrace(pyutilib.misc.traceit)
    unittest.main()
