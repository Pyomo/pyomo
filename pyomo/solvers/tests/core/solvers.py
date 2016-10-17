
__all__ = ['test_solvers']

import re
import os
import sys
import time
import csv
from os.path import abspath, dirname
import logging
import pprint

from pyutilib.misc import Bunch, Options
import pyutilib.th as unittest
import pyutilib.autotest
import pyomo.misc.plugin

import pyomo.solvers.plugins.testdriver.pyomo
#import pyomo.data.pyomo.plugins

currdir = dirname(abspath(__file__))+os.sep
rootdir = None


def percentParts(numer, denom):
    if denom == 0:
        return "0"
    else:
        return int(round(100.0 * float(numer) / float(denom)))

def percentStrParts(numer, denom):
    return str(percentParts(numer, denom))+'%'


class SolverResultPrinter:

    def __init__(self, options):
        self.options = options

    def group_data(self, solverData, testResult):
        """
        Group the data by the solver that produced it.

        @param solverData The 'data' object produced during solver testing;
                          contains information for solvers that passed their
                          unit tests.
        @param testResult The result object produced by the unittest framework.
        @return A dictionary mapping solver names to the elements in solverData.
        """

        ans = {}
        solverNameRe = re.compile("test_([a-zA-Z0-9_]*)_TEST")

        for test in testResult.runner.suite._tests:
            match = solverNameRe.match(test._testMethodName)
            if match is not None:
                solver = match.group(1)
                if solver not in ans:
                    ans[solver] = [x for x in solverData if x.name.startswith(solver+'_TEST')]

        return ans

    def write(self, solverData, testResult, filename):
        odir = os.getcwd()
        os.chdir(rootdir)
        with open(filename, 'w') as OUTPUT:
            print("\nWriting results summary to file '%s'" % abspath(filename))
            self.pprint(solverData, testResult, stream=OUTPUT)
            OUTPUT.close()
        os.chdir(odir)

    def pprint(self, solverData, testResult, stream = sys.stdout):
        """
        Print the results of solver testing in a human-readable format.

        This method is to be overridden by subclasses of SolverResultPrinter
        to provide different formats of printout, depending on the user's
        choices (given by command-line arguments).
        """
        raise Exception("Unimplemented")


class TabularResultPrinter(SolverResultPrinter):

    def pprint(self, solverData, testResult, stream = sys.stdout):
        groups = self.group_data(solverData, testResult)

        maxSolverNameLen = max([max(len(name) for name in groups), len("Solver")])
        fmtStr = "{{0:<{0}}}| {{1:>8}} | {{2:>8}} | {{3:>6}} | {{4:>13}}\n".format(maxSolverNameLen + 2)

        stream.write("\n")
        stream.write("Solver Check Test Summary\n")
        stream.write("=" * (maxSolverNameLen + 49) + "\n")
        stream.write(fmtStr.format("Solver", "# Run", "# Pass", "% Pass", "Mean '% Pass'"))
        stream.write("=" * (maxSolverNameLen + 49) + "\n")

        allPass = 0
        allRun = 0
        for solver in sorted(groups.keys()):
            totalPass = 0
            totalRun = 0
            norm = {}
            for item in groups[solver]:
                if not item.problem in norm:
                    norm[item.problem] = []
                if item.result[0] is True:
                    norm[item.problem].append(1)
                    totalPass += 1
                else:
                    norm[item.problem].append(0)
                totalRun += 1
            if totalRun > 0:
                norm_val = 0.0
                for item in norm:
                    norm_val += sum(norm[item])/(1.0*len(norm[item]))
                stream.write(fmtStr.format(solver, totalRun, totalPass, percentStrParts(totalPass, totalRun), "%10.2f" % norm_val))
                allPass += totalPass
                allRun += totalRun
            else:
                stream.write(fmtStr.format(solver, 0, 0, percentStrParts(0,0), "%10.2f" % 0.0))
        stream.write("-" * (maxSolverNameLen + 49) + "\n")
        stream.write(fmtStr.format("Total", allRun, allPass, percentStrParts(allPass, allRun), "%10.2f" % len(norm)) + "\n")


class VerboseResultPrinter(SolverResultPrinter):

    def pprint(self, solverData, testResult, stream = sys.stdout):
        groups = self.group_data(solverData, testResult)

        for solver in groups:
            stream.write("\n")
            stream.write("=== Solver: {0} ===\n".format(solver))
            if len(groups[solver]) == 0:
                stream.write('  No results\n')
            else:
                for item in groups[solver]:
                    if self.options.debug:
                        if item.result[0] is True:
                            result = 'PASS: '+item.result[1]
                        else:
                            result = 'FAIL: '+item.result[1]
                    elif item.result[0] is False:
                        result = 'FAIL: '+item.result[1]
                    else:
                        continue
                    stream.write('  - suite={0}\n    problem={1}\n    result="{2}"\n'.format(item.suite, item.problem, result))


class CSVResultPrinter(SolverResultPrinter):

    def pprint(self, solverData, testResult, stream = sys.stdout):
        groups = self.group_data(solverData, testResult)

        if not os.environ.get('HUDSON_URL',None) is None:
            colprefix = ['job','build','node']
            prefix = [os.environ['JOB_NAME'], os.environ['BUILD_NUMBER'], os.environ['NODE_NAME']]
        else:
            colprefix = []
            prefix=[]

        ans = [colprefix + ['classname', 'name', 'dataname', 'value']]
        allPass = 0
        allRun = 0
        for solver in groups:
            totalPass = 0
            totalRun = 0
            for item in groups[solver]:
                if item.result[0] is True:
                    totalPass += 1
                totalRun += 1
            if totalRun > 0:
                ans.append( prefix+['pyomo.test-solvers', solver, '% pass', percentParts(totalPass, totalRun)] )
                allPass += totalPass
                allRun += totalRun
        ans.append( prefix+['pyomo.test-solvers', 'AllSolvers', '% pass', percentParts(allPass, allRun)] )

        writer = csv.writer(stream)
        writer.writerows(ans)

class SolverTestRunner(unittest.TextTestRunner):

    class NullStream:

        def write(self, str):
            pass

        def flush(self):
            pass


    class SolverTestResult(unittest.TestResult):

        def __init__(self, runner):
            unittest.TestResult.__init__(self)
            self.runner = runner


    def __init__(self):
        unittest.TextTestRunner.__init__(self, verbosity=2)

    def run(self, test):
        # Prep for test run
        result = SolverTestRunner.SolverTestResult(self)
        testset = reduce(set.union, [set(t._tests) for t in test._tests], set())
        filteredSuite = unittest.TestSuite()
        if len(self.options.solver) == 0:
            prefix = re.compile("test_")
        else:
            prefix = re.compile("test_({0})_TEST".format("|".join(self.options.solver)))
        filteredSet = list( filter((lambda x : prefix.match(str(x._testMethodName)) is not None), testset) )
        filteredSet.sort()
        filteredSuite.addTests(filteredSet)

        logging.disable(logging.CRITICAL)
        # Temporarily disable printing
        if not self.options.debug:
            sys.stdout = SolverTestRunner.NullStream()
            sys.stderr = SolverTestRunner.NullStream()

        # Perform the run
        self.suite = filteredSuite
        filteredSuite(result)

        # Re-enable logging, printing
        if not self.options.debug:
            #logging.disable(logging.NOTSET)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        # Report results
        print("\nTest Problem Summary - {0} problems: {1} failures, {2} errors ({3} skipped)".format(result.testsRun, len(result.failures), len(result.errors), len(result.skipped)))
        sys.stderr.flush()
        sys.stdout.flush()

        # Info about data
        TabularResultPrinter(self.options).pprint(SolverTestingDriver.data, result)
        if self.options.verbose:
            VerboseResultPrinter(self.options).pprint(SolverTestingDriver.data, result)
        if not self.options.csv is None:
            CSVResultPrinter(self.options).write(SolverTestingDriver.data, result, self.options.csv)
        return result


@unittest.nottest
def test_solvers(options=None, argv=None):
    """
    The is the function executed by the command
        pyomo test-solvers [solver ...]
    """
    global rootdir
    rootdir = os.getcwd()
    if argv is None:
        if options.debug:
            if len(options.solver) == 0:
                print("Testing all solvers")
            else:
                print("Testing solver", options.solver[0])
        # Over-ride the value of sys.argv, which is used by unittest.main()
        sys.argv=['test_solver']
    else:
        sys.argv=argv
    # Create the tests defined in the YAML configuration file
    autotest_options = Options()
    autotest_options.testname_format = "%s_TEST_%s"
    pyutilib.autotest.create_test_suites(filename=currdir+'test_solvers.yml', _globals=globals(), options=autotest_options)
    # Execute the tests, using a custom test runner
    runner = SolverTestRunner()
    runner.options = options
    unittest.main(module=globals()['__name__'], testRunner=runner)


class SolverTestingDriver(pyomo.solvers.plugins.testdriver.pyomo.PyomoTestDriver):
    """
    A test driver plugin that is used to test solvers.
    """

    pyomo.misc.plugin.alias('pyomo.test_solvers')

    data = []

    def run_test(self, testcase, name, options):
        sys.stdout.write('\n')
        sys.stdout.flush()
        # Setup option values that are specific to this test driver
        options.problemdir = options.currdir+'problems'+os.sep
        options.model = options.problemdir + options.problem+'.py'
        # Set the tester to work in the specific problem directory
        sys.path.append(options.problemdir)
        odir = os.getcwd()
        os.chdir(options.problemdir)
        # Add additional options to the pyomo commandline
        ##options.pyomo = "-v --stream-solver --solver-suffixes=all"
        options.pyomo = "-v --stream-solver"
        # Call the run_test method in PyomoTestDriver
        try:
            pyomo.solvers.plugins.testdriver.pyomo.PyomoTestDriver.run_test(self, testcase, name, options)
            try:
                self.perform_tests_locally(testcase, name, options)
            except Exception:
                e = sys.exc_info()[1]
                print("ERROR: "+str(e))
                sys.path.remove(options.problemdir)
                testcase.fail(str(e))
        except Exception:
            e = sys.exc_info()[1]
            msg = 'Pyomo Execution Error: '+str(e)
            # record total check data
            self.recordData(testcase, name, options, {'error': (False, 'No checks performed')})
            sys.path.remove(options.problemdir)
            testcase.fail(msg)
        sys.path.remove(options.problemdir)
        os.chdir(odir)

    def recordData(self, testcase, name, options, data):
        """Record all data in the 'data' dictionary"""
        for key in data:
            self.data.append( Bunch(name=name, problem=options.problem, suite=options.suite, test=key, result=data[key]))

    def perform_tests(self, testcase, name, options):
        """
        Over-ride the tests that are performed by the Pyomo test driver.
        """
        pass

    def perform_tests_locally(self, testcase, name, options):
        """
        Over-ride the tests that are performed by the Pyomo test driver.
        """
        import pyomo.opt


        if options.tests is None:
            raise IOError("No 'tests' option for test '%s'" % options.problem)
        if not options.baseline is None:
            baseline = pyomo.opt.SolverResults()
            try:
                baseline.read(filename=options.baseline)
            except Exception:
                e = sys.exc_info()[1]
                s = "ERROR: Problem reading solver results in file '{0}': {1}".format(options.baseline, str(e))
                raise IOError(s)
            options._baseline = baseline
        data = {}
        # For each test in options. test, call the corresponding plugin function
        try:
            for test in re.split('[ ]+',options.tests):
                # WEH - Edit this to figure out which packages we are running
                #fn = getattr(pyomo.data.pyomo.plugins, test)
                fn(options.results, data, options)
        except Exception:
            e = sys.exc_info()[1]
            s = "ERROR: Problem performing solver checks in {2} for {0}: {1}".format(name, str(e), test)
            raise IOError(s)
        # compute total check data
        if len(data) == 0:
            data['error'] = False, 'No checks performed'
        # record total check data
        self.recordData(testcase, name, options, data)
        # Summarize all check failures
        num = sum(1 for key in data if data[key][0] == False)
        if num > 0:
            print("Solver Check Failures: %s" % name)
            for key in data:
                if not data[key][0]:
                    print(" %s" % str(data[key][1]))
        else:
            print("No Solver Check Failures: %s" % name)

