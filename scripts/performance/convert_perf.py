#
# This script runs performance tests while converting a model
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


_bin = {'COOPR3': '/Users/wehart/src/pyomo/py36/bin',
        'PYOMO5': os.path.abspath('../../../../bin'),
        'PYPY': '/Users/wehart/src/pyomo/pypy/bin'
       }
#exdir = '../../../pyomo_prod/examples/performance'
exdir = '../../examples/performance'

large = True
_timeout = 20
#N = 30
N = 1


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="Save results to the specified file", action="store", default=None)
parser.add_argument("-t", "--type", help="Specify the file type to test", action="store", default=None)
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
                if tokens[3:5] == ['to', 'construct']:
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
# Convert a test problem
#
def run_pyomo(code, format_, problem, verbose, cwd=None):

    if verbose:
        options = ""  # TODO
    else:
        options = ""

    def f():
        cmd = _bin[code] + '/pyomo convert --report-timing --output=file.%s %s %s' % (format_, options, problem)
        _cwd = os.getcwd()
        if not cwd is None:
            os.chdir(cwd)
        if verbose:
            print("Command: %s" % cmd)
        res = pyutilib.subprocess.run(cmd, outfile='pyomo.out', verbose=verbose)
        if res[0] != 0:
            print("Aborting performance testing!")
            sys.exit(1)

        seconds = {}
        eval_ = evaluate('pyomo.out', seconds, verbose)
        os.chdir(_cwd)
        return eval_

    return f

#
# Convert a test problem
#
def run_script(code, format_, problem, verbose, cwd=None):

    if verbose:
        options = ""  # TODO
    else:
        options = ""

    def f():
        cmd = _bin[code] + '/lpython %s pyomo.%s' % (problem, format_)
        if verbose:
            print("Command: %s" % cmd)
        _cwd = os.getcwd()
        os.chdir(cwd)
        res = pyutilib.subprocess.run(cmd, outfile='pyomo.out', verbose=verbose)
        os.chdir(_cwd)
        if res[0] != 0:
            print("Aborting performance testing!")
            sys.exit(1)

        seconds = {}
        return evaluate(cwd+'/pyomo.out', seconds, verbose)

    return f


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
def runall(factors, res, output=True, filetype=None, verbose=False):

    code = factors[0]

    def pmedian1(name, num):
        testname = 'pmedian1_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/pmedian/pmedian1.py %s/pmedian/pmedian.test%d.dat" % (exdir, exdir, num), verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def pmedian2(name, num):
        testname = 'pmedian2_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/pmedian/pmedian2.py %s/pmedian/pmedian.test%d.dat" % (exdir, exdir, num), verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def bilinear1(name, num):
        testname = 'bilinear1_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/misc/bilinear1_%d.py" % (exdir, num), verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def bilinear2(name, num):
        testname = 'bilinear2_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/misc/bilinear2_%d.py" % (exdir, num), verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def diag1(name, num):
        testname = 'diag1_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/misc/diag1_%d.py" % (exdir, num), verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def diag2(name, num):
        testname = 'diag2_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/misc/diag2_%d.py" % (exdir, num), verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def stochpdegas1(name, num):
        testname = 'stochpdegas1_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_script(code, name, "run_stochpdegas1_automatic.py", verbose, cwd='%s/dae/' % exdir), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def dcopf1(name, num):
        testname = 'dcopf1_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_script(code, name, "perf_test_dcopf_case2383wp.py", verbose, cwd='%s/dcopf/' % exdir), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def uc1(name, num):
        testname = 'uc1_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/uc/ReferenceModel.py %s/uc/2014-09-01-expected.dat" % (exdir, exdir), verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def jump_clnlbeam(name, num):
        testname = 'jump_clnlbeam_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/jump/clnlbeam.py %s/jump/clnlbeam-%d.dat" % (exdir, exdir, num), verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def jump_facility(name, num):
        testname = 'jump_facility'
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/jump/facility.py" % exdir, verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def jump_lqcp(name, num):
        testname = 'jump_lqcp'
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "%s/jump/lqcp.py" % exdir, verbose), n=N)
            if not verbose:
                print_results(factors_, ans_, output)

    def jump_opf(name, num):
        testname = 'jump_opf_%d' % num
        if not filetype or filetype == name:
            factors_ = tuple(factors+[name,testname])
            print("TESTING: %s" % " ".join(factors_))
            ans_ = res[factors_] = measure(run_pyomo(code, name, "opf_%dbus.py" % num, verbose, cwd="%s/jump" % exdir), n=N)
            if not verbose:
                print_results(factors_, ans_, output)


    if True:
        #jump_clnlbeam('nl', 500000)
        #jump_opf('nl', 66200)
        if large:
            jump_clnlbeam('nl', 50000)
            jump_opf('nl', 6620)
        else:
            jump_clnlbeam('nl', 5000)
            jump_opf('nl', 662)
        jump_facility('nl', 0)
        jump_lqcp('nl', 0)

    if True:
        if large:
            dcopf1('lp', 0)
            dcopf1('nl', 0)
        else:
            dcopf1('lp', 0)
            dcopf1('nl', 0)

    if True:
        if large:
            stochpdegas1('nl', 0)
        else:
            stochpdegas1('nl', 0)

    if True:
        if large:
            uc1('lp', 0)
            uc1('nl', 0)
        else:
            uc1('lp', 0)
            uc1('nl', 0)

    if True:
        if large:
            pmedian1('lp',8)
            pmedian1('nl',8)
        else:
            pmedian1('lp',4)
            pmedian1('nl',4)

    if True:
        if large:
            pmedian2('lp',8)
            pmedian2('nl',8)
        else:
            pmedian2('lp',4)
            pmedian2('nl',4)

    if True:
        if large:
            bilinear1('lp',100000)
            bilinear1('nl',100000)
        else:
            bilinear1('lp',100)
            bilinear1('nl',100)

    if True:
        if large:
            bilinear2('lp',100000)
            bilinear2('nl',100000)
        else:
            bilinear2('lp',100)
            bilinear2('nl',100)

    if True:
        if large:
            diag1('lp',100000)
            diag1('nl',100000)
        else:
            diag1('lp',100)
            diag1('nl',100)

    if True:
        if large:
            diag2('lp',100000)
            diag2('nl',100000)
        else:
            diag2('lp',100)
            diag2('nl',100)


def remap_keys(mapping):
    return [{'factors':k, 'performance': v} for k, v in mapping.items()]

#
# MAIN
#
res = {}

runall(['COOPR3'], res, filetype=args.type, verbose=args.verbose)
runall(['PYOMO5'], res, filetype=args.type, verbose=args.verbose)
runall(['PYPY'], res, filetype=args.type, verbose=args.verbose)



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

