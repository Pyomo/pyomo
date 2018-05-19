#
# This script runs performance tests on solvers
#

from pyomo.environ import *
import pyomo.version
from pyomo.core.base.expr_common import _clear_expression_pool
from pyomo.core.base import expr as EXPR

import pprint as pp
import gc
import time
try:
    import pympler
    pympler_available=True
    pympler_kwds = {}
except:
    pympler_available=False
import sys
import argparse
import re
import pyutilib.subprocess

## TIMEOUT LOGIC
from functools import wraps
import errno
import os
import signal


class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=10, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)



_timeout = 20
#N = 30
N = 1


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="Save results to the specified file", action="store", default=None)
#parser.add_argument("-s", "--solver", help="Specify the solver to test", action="store", default=None)
parser.add_argument("-v", "--verbose", help="Run just a single trial in verbose mode", action="store_true", default=False)
parser.add_argument("--ntrials", help="The number of test trials", action="store", type=int, default=None)
args = parser.parse_args()

if args.ntrials:
    N = args.ntrials
print("NTrials %d\n\n" % N)


#
# Execute a function 'n' times, collecting performance statistics and
# averaging them
#
def measure(f, n=25):
    """measure average execution time over n trials"""
    data = []
    for i in range(n):
        data.append(f())
        sys.stdout.write('.')
        sys.stdout.flush()
    sys.stdout.write('\n')
    #
    ans = {}
    for key in data[0]:
        d_ = []
        for i in range(n):
            d_.append( data[i][key] )
        ans[key] = {"mean": sum(d_)/float(n), "data": d_}
    #
    return ans


#
# Evaluate Pyomo output
#
def evaluate(logfile, seconds, verbose):
    with open(logfile, 'r') as OUTPUT:
        if verbose:
            sys.stdout.write("*" * 50 + "\n")

        for line in OUTPUT:
            if verbose:
                sys.stdout.write(line)
            tokens = re.split('[ \t]+', line.strip())
            #print(tokens)
            if len(tokens) < 2:
                pass
            elif tokens[1] == 'seconds' and tokens[2] == 'required':
                if tokens[3:6] == ['to', 'construct', 'instance=unknown']:
                    seconds['construct'] = float(tokens[0])
                elif tokens[3:6] == ['to', 'write', 'file']:
                    seconds['write_problem'] = float(tokens[0])
                elif tokens[3:6] == ['to', 'read', 'logfile']:
                    seconds['read_logfile'] = float(tokens[0])
                elif tokens[3:6] == ['to', 'read', 'solution']:
                    seconds['read_solution'] = float(tokens[0])
                elif tokens[3:5] == ['for', 'solver']:
                    seconds['solver'] = float(tokens[0])
                elif tokens[3:5] == ['for', 'presolve']:
                    seconds['presolve'] = float(tokens[0])
                elif tokens[3:5] == ['for', 'postsolve']:
                    seconds['postsolve'] = float(tokens[0])
                elif tokens[3:6] == ['for', 'problem', 'transformations']:
                    seconds['transformations'] = float(tokens[0])

        if verbose:
            sys.stdout.write("*" * 50 + "\n")
    return seconds


#
# Solve a test problem
#
def run_pyomo(solver, problem, verbose):

    if verbose:
        options = "--stream-solver"
    else:
        options = ""

    def f():
        cmd = 'pyomo solve --solver=%s --report-timing --results-format=json --save-results=solver.jsn %s %s' % (solver, options, problem)
        if verbose:
            print("Command: %s" % cmd)
        res = pyutilib.subprocess.run(cmd, outfile='solver.out')
        if res[0] != 0:
            print("Aborting performance testing!")
            sys.exit(1)

        seconds = {}
        return evaluate('solver.out', seconds, verbose)

    return f


def solve_pmedian(solver, num, verbose):
    return run_pyomo(solver, "../../examples/performance/pmedian/pmedian.py ../../examples/performance/pmedian/pmedian.test%d.dat" % num, verbose)

#
# Utility function used by runall()
#
def print_results(factors_, ans_, output):
    if output:
        print(factors_)
        pp.pprint(ans_)
        print("")


#
# Run the experiments and populate the dictionary 'res' 
# with the mapping: factors -> performance results
#
# Performance results are a mapping: name -> seconds
#
def runall(factors, res, output=True, num=4, solver=None, verbose=False):

    testname = 'pmedian%d' % num
    def runone(name):
        if not solver or solver == name:
            if pyomo.opt.check_available_solvers(name):
                factors_ = tuple(factors+[name,testname])
                print("TESTING: %s" % " ".join(factors_))
                ans_ = res[factors_] = measure(solve_pmedian(name, num, verbose=verbose), n=N, verbose=verbose)
                if not verbose:
                    print_results(factors_, ans_, output)
            else:
                print("Missing %s\n" % name)


    runone('cplex')
    runone('_cplex_direct')
    runone('_cplex_persistent')
    runone('gurobi')
    runone('_gurobi_direct')
    runone('scip')
    runone('xpress')
    runone('glpk')
    runone('cbc')
    runone('ipopt')
    if False:
        runone('baron')



def remap_keys(mapping):
    return [{'factors':k, 'performance': v} for k, v in mapping.items()]

#
# MAIN
#
res = {}

runall([], res, num=4, solver=args.solver, verbose=args.verbose)



if args.output:
    if args.output.endswith(".csv"):
        #
        # Write csv file
        #
        perf_types = sorted(next(iter(res.values())).keys())
        res_ = [ list(key) + [res.get(key,{}).get(k,{}).get('mean',-777) for k in perf_types] for key in sorted(res.keys())]
        with open(args.output, 'w') as OUTPUT:
            import csv
            writer = csv.writer(OUTPUT)
            writer.writerow(['Version', 'ExprType', 'ExprNum'] + perf_types)
            for line in res_:
                writer.writerow(line)

    elif args.output.endswith(".json"):
        res_ = {'script': sys.argv[0], 'NTrials':N, 'data': remap_keys(res), 'pyomo_version':pyomo.version.version, 'pyomo_versioninfo':pyomo.version.version_info[:3]}
        #
        # Write json file
        #
        with open(args.output, 'w') as OUTPUT:
            import json
            json.dump(res_, OUTPUT)

    else:
        print("Unknown output format for file '%s'" % args.output)

