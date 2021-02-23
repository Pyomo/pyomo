#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import json
import os
from nose.tools import nottest

import pyomo.common.unittest as unittest

import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers

currdir = os.path.dirname(os.path.abspath(__file__))
exdir = os.path.abspath(os.path.join(currdir, '..', '..', '..', '..', 'examples', 'pyomo', 'diet'))

sqlite3_available = pyodbc_available = False
try:
    import sqlite3
    sqlite3_available = True
except ImportError:
    pass
try:
    import pyodbc
    pyodbc_available = True
    #
    # Temporarily deprecating pyodbc tests.
    # These tests are not reliably executing with Python 2.6 and 2.7, 
    # due to apparent issues with unicode representation.
    #
    pyodbc_available = False
except ImportError:
    pass

solvers = None
class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = check_available_solvers('glpk')

    @nottest
    def run_pyomo(self, *args, **kwargs):
        """
        Run Pyomo with the given arguments. `args` should be a list with
        one argument token per string item.
        """
        if not 'glpk' in solvers:
            self.skipTest("GLPK is not installed")

        if isinstance(args, str):
            args = args.split()
        args = list(map(str, args))

        outputpath = kwargs.pop('outputpath', os.path.join(exdir, 'results.jsn'))
        args = ['solve', '--solver=glpk', '--results-format=json', '-c', '--logging=quiet', '--save-results', outputpath] + args

        old_path = os.getcwd()
        os.chdir(exdir)
        output = main.main(args)
        os.chdir(old_path)

        return outputpath

    def test_pyomo_dat(self):
        results_file = self.run_pyomo(os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet.dat'), outputpath=os.path.join(currdir, 'pyomo_dat.jsn'))
        baseline_file = os.path.join(currdir, 'baselines', 'diet1_pyomo_dat.jsn')
        self.assertMatchesJsonBaseline(results_file, baseline_file)

    @unittest.category('nightly')
    @unittest.skipUnless(pyodbc_available, "Requires PyODBC")
    def test_pyomo_mdb(self):
        results_file = self.run_pyomo(os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet1.db.dat'), outputpath=os.path.join(currdir, 'pyomo_mdb.jsn'))
        baseline_file = os.path.join(currdir, 'baselines', 'diet1_pyomo_mdb.jsn')
        self.assertMatchesJsonBaseline(results_file, baseline_file)

    @unittest.category('nightly')
    @unittest.skipUnless(pyodbc_available, "Requires PyODBC")
    def test_mdb_equality(self):
        dat_results_file = self.run_pyomo(
            os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet.dat'),
            outputpath=os.path.join(currdir, 'dat_results.jsn'))
        with open(dat_results_file) as FILE:
            dat_results = json.load(FILE)
        db_results_file = self.run_pyomo(
            os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet1.db.dat'),
            outputpath=os.path.join(currdir, 'db_results.jsn'))
        with open(db_results_file) as FILE:
            db_results = json.load(FILE)
        # Filter out the solver time
        del dat_results['Solver'][0]['Time']
        del db_results['Solver'][0]['Time']
        # Compare baselines
        self.assertStructuredAlmostEqual(dat_results, db_results)
        os.remove(dat_results_file)
        os.remove(db_results_file)

    @unittest.skipUnless(sqlite3_available, "Requires SQLite3")
    def test_pyomo_sqlite3(self):
        results_file = self.run_pyomo(os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet1.sqlite.dat'), outputpath=os.path.join(currdir, 'pyomo_sqlite3.jsn'))
        baseline_file = os.path.join(currdir, 'baselines', 'diet1_pyomo_sqlite3.jsn')
        self.assertMatchesJsonBaseline(results_file, baseline_file)

    @unittest.skipUnless(sqlite3_available, "Requires SQLite3")
    def test_sqlite_equality(self):
        dat_results_file = self.run_pyomo(
            os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet.dat'),
            outputpath=os.path.join(currdir, 'dat_results.jsn'))
        with open(dat_results_file) as FILE:
            dat_results = json.load(FILE)
        sqlite_results_file = self.run_pyomo(
            os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet1.sqlite.dat'),
            outputpath=os.path.join(currdir, 'sqlite_results.jsn'))
        with open(sqlite_results_file) as FILE:
            sqlite_results = json.load(FILE)
        # Filter out the solver time
        del dat_results['Solver'][0]['Time']
        del sqlite_results['Solver'][0]['Time']
        # Compare baselines
        self.assertStructuredAlmostEqual(dat_results, sqlite_results)
        os.remove(dat_results_file)
        os.remove(sqlite_results_file)

if __name__ == "__main__":
    unittest.main()

