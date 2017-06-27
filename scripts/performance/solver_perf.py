#
# This script runs performance tests on solvers
#

from pyomo.environ import *
import pyutilib.subprocess

import re
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
import getopt

#N = 50
N = 5


try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "hc:", ["help", "output=", 'num='])
except getopt.GetoptError as err:
    # print help information and exit:
    print(str(err))  # will print something like "option -a not recognized"
    print(sys.argv[0] + " -h -c --num=<ntrials> --output=<filename>")
    sys.exit(2)

ofile = None
file_format = 'csv'
for o, a in opts:
    if o in ("-h", "--help"):
        print(sys.argv[0] + " -c -h --num=<ntrials> --output=<filename>")
        sys.exit()
    elif o == "--output":
        ofile = a
    elif o == "--num":
        N = int(a)
    elif o == "-c":
        print(a)
        file_format = 'csv'
    else:
        assert False, "unhandled option"

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
    sys.stdout.write('\n')
    #
    ans = {}
    for key in data[0]:
        total = 0
        for i in range(n):
            total += data[i][key]
        ans[key] = total/float(n)
    #
    return ans



#
# Evaluate Pyomo output
#
def evaluate(logfile, seconds):
    with open(logfile, 'r') as OUTPUT:
        for line in OUTPUT:
            tokens = re.split('[ \t]+', line.strip())
            #print(tokens)
            if tokens[1] == 'seconds' and tokens[2] == 'required':
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

    return seconds


#
# Solve a test problem
#
def run_pyomo(solver, problem):

    def f():
        res = pyutilib.subprocess.run('pyomo solve --solver=%s --report-timing --results-format=json --save-results=solver.jsn %s' % (solver, problem), outfile='solver.out')

        seconds = {}
        return evaluate('solver.out', seconds)

    return f


def solve_pmedian6(solver):
    return run_pyomo(solver, "../../examples/pyomo/p-median/pmedian.py ../../examples/pyomo/p-median/pmedian.test6.dat")

def solve_pmedian7(solver):
    return run_pyomo(solver, "../../examples/pyomo/p-median/pmedian.py ../../examples/pyomo/p-median/pmedian.test7.dat")

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
def runall(factors, res, output=True):

    factors_ = tuple(factors+['cbc','pmedian6'])
    ans_ = res[factors_] = measure(solve_pmedian6('cbc'), n=N)
    print_results(factors_, ans_, output)

    factors_ = tuple(factors+['ipopt','pmedian6'])
    ans_ = res[factors_] = measure(solve_pmedian6('ipopt'), n=N)
    print_results(factors_, ans_, output)

    #factors_ = tuple(factors+['cbc','pmedian7'])
    #ans_ = res[factors_] = measure(solve_pmedian7('cbc'), n=N)
    #print_results(factors_, ans_, output)


def remap_keys(mapping):
    return [{'factors':k, 'performance': v} for k, v in mapping.items()]

#
# MAIN
#
res = {}

runall([], res)



if ofile:
    if file_format == 'csv':
        #
        # Write csv file
        #
        perf_types = sorted(next(iter(res.values())).keys())
        res_ = [ list(key) + [res[key][k] for k in perf_types] for key in res]
        with open(ofile, 'w') as OUTPUT:
            import csv
            writer = csv.writer(OUTPUT)
            writer.writerow(['Version', 'ExprType', 'ExprNum'] + perf_types)
            for line in res_:
                writer.writerow(line)

    elif file_format == 'json':
        res_ = {'script': sys.argv[0], 'NTrials':N, 'data': remap_keys(res)}
        #
        # Write json file
        #
        with open(ofile, 'w') as OUTPUT:
            import json
            json.dump(res_, OUTPUT)
