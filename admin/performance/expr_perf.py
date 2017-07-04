#
# This script runs performance tests on expressions
#

from pyomo.environ import *
import pyomo.version
from pyomo.core.base.expr_common import _clear_expression_pool
from pyomo.core.base import expr as EXPR 
from pyomo.repn import generate_canonical_repn

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

sys.setrecursionlimit(1000000)
#NTerms = 100000
#N = 50
NTerms = 100
N = 5


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="Save results to the specified file", action="store", default=None)
parser.add_argument("--nterms", help="The number of terms in test expressions", action="store", type=int, default=None)
parser.add_argument("--ntrials", help="The number of test trials", action="store", type=int, default=None)
args = parser.parse_args()

if args.nterms:
    NTerms = args.nterms
if args.ntrials:
    N = args.ntrials
print("NTerms %d   NTrials %d\n\n" % (NTerms, N))



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
# Evaluate standard operations on an expression
#
def evaluate(expr, seconds):
    if False:
        gc.collect()
        _clear_expression_pool()
        start = time.time()
        #
        expr_ = expr.clone()
        #
        stop = time.time()
        seconds['clone'] = stop-start

    gc.collect()
    _clear_expression_pool()
    start = time.time()
    #
    d_ = expr.polynomial_degree()
    #
    stop = time.time()
    seconds['polynomial_degree'] = stop-start

    if False:
        gc.collect()
        _clear_expression_pool()
        start = time.time()
        #
        s_ = expr.to_string()
        #
        stop = time.time()
        seconds['to_string'] = stop-start

    gc.collect()
    _clear_expression_pool()
    start = time.time()
    #
    s_ = expr.is_constant()
    #
    stop = time.time()
    seconds['is_constant'] = stop-start

    gc.collect()
    _clear_expression_pool()
    start = time.time()
    #
    s_ = expr.is_fixed()
    #
    stop = time.time()
    seconds['is_fixed'] = stop-start

    try:
        gc.collect()
        _clear_expression_pool()
        start = time.time()
        #
        r_ = generate_canonical_repn(expr)
        #
        stop = time.time()
        seconds['generate_canonical'] = stop-start
    except:
        seconds['generate_canonical'] = -1

    gc.collect()
    _clear_expression_pool()
    start = time.time()
    #
    s_ = EXPR.compress_expression(expr)
    #
    stop = time.time()
    seconds['compress'] = stop-start

    return seconds


#
# Create a linear expression
#
def linear(N, flag):

    def f():
        seconds = {}

        model = ConcreteModel()
        model.A = RangeSet(N)
        model.p = Param(model.A, default=2)
        model.x = Var(model.A, initialize=2)

        for i in model.A:
            if i != N:
                model.x[i].fixed = True

        gc.collect()
        _clear_expression_pool()
        start = time.time()
        #
        if flag == 1:
            expr = summation(model.p, model.x)
        elif flag == 2:
            expr=sum(model.p[i]*model.x[i] for i in model.A)
        else:
            expr=0
            for i in model.A:
                expr += model.p[i] * model.x[i]
        #
        stop = time.time()
        seconds['construction'] = stop-start

        #import pdb;  pdb.set_trace()
        seconds = evaluate(expr, seconds)

        return seconds

    return f


#
# Create a constant expression from mutable parameters
#
def constant(N, flag):

    def f():
        seconds = {}

        model = ConcreteModel()
        model.A = RangeSet(N)
        model.p = Param(model.A, default=2)
        model.x = Param(model.A, initialize=2, mutable=True)

        gc.collect()
        _clear_expression_pool()
        start = time.time()
        #
        if flag == 1:
            expr = summation(model.p, model.x, index=model.A)
        elif flag == 2:
            expr=sum(model.p[i]*model.x[i] for i in model.A)
        else:
            expr=0
            for i in model.A:
                expr += model.p[i] * model.x[i]
        #
        stop = time.time()
        seconds['construction'] = stop-start

        seconds = evaluate(expr, seconds)

        return seconds

    return f


#
# Create a bilinear expression
#
def bilinear(N, flag):

    def f():
        seconds = {}

        model = ConcreteModel()
        model.A = RangeSet(N)
        model.p = Param(model.A, default=2)
        model.x = Var(model.A, initialize=2)
        model.y = Var(model.A, initialize=2)

        for i in model.A:
            if i != N:
                model.x[i].fixed = True
                model.y[i].fixed = True

        gc.collect()
        _clear_expression_pool()
        start = time.time()
        #
        if flag == 1:
            expr = summation(model.p, model.x, model.y)
        elif flag == 2:
            expr=sum(model.p[i]*model.x[i]*model.y[i] for i in model.A)
        else:
            expr=0
            for i in model.A:
                expr += model.p[i] * model.x[i] * model.y[i]
        #
        stop = time.time()
        seconds['construction'] = stop-start

        seconds = evaluate(expr, seconds)

        return seconds

    return f


#
# Create a simple nonlinear expression
#
def nonlinear(N, flag):

    def f():
        seconds = {}

        model = ConcreteModel()
        model.A = RangeSet(N)
        model.p = Param(model.A, default=2)
        model.x = Var(model.A, initialize=2)

        for i in model.A:
            if i != N:
                model.x[i].fixed = True

        gc.collect()
        _clear_expression_pool()
        start = time.time()
        #
        if flag == 2:
            expr=sum(model.p[i]*tan(model.x[i]) for i in model.A)
        else:
            expr=0
            for i in model.A:
                expr += model.p[i] * tan(model.x[i])
        #
        stop = time.time()
        seconds['construction'] = stop-start

        seconds = evaluate(expr, seconds)

        return seconds

    return f


#
# Create an expression that is a complex polynomial
#
def polynomial(N, flag):

    def f():
        seconds = {}

        model = ConcreteModel()
        model.A = RangeSet(N)
        model.p = Param(model.A, default=2)
        model.x = Var(model.A, initialize=2)

        for i in model.A:
            if i != N:
                model.x[i].fixed = True

        gc.collect()
        _clear_expression_pool()
        start = time.time()
        #
        if True:
            expr=0
            for i in model.A:
                expr = model.x[i] * (1 + expr)
        #
        stop = time.time()
        seconds['construction'] = stop-start

        seconds = evaluate(expr, seconds)

        return seconds

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
def runall(factors, res, output=True):

    if False:
        factors_ = tuple(factors+['Constant','Loop 1'])
        ans_ = res[factors_] = measure(constant(NTerms, 1), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Constant','Loop 2'])
        ans_ = res[factors_] = measure(constant(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Constant','Loop 3'])
        ans_ = res[factors_] = measure(constant(NTerms, 3), n=N)
        print_results(factors_, ans_, output)


        factors_ = tuple(factors+['Linear','Loop 1'])
        ans_ = res[factors_] = measure(linear(NTerms, 1), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 2'])
        ans_ = res[factors_] = measure(linear(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 3'])
        ans_ = res[factors_] = measure(linear(NTerms, 3), n=N)
        print_results(factors_, ans_, output)


        factors_ = tuple(factors+['Bilinear','Loop 1'])
        ans_ = res[factors_] = measure(bilinear(NTerms, 1), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Bilinear','Loop 2'])
        ans_ = res[factors_] = measure(bilinear(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Bilinear','Loop 3'])
        ans_ = res[factors_] = measure(bilinear(NTerms, 3), n=N)
        print_results(factors_, ans_, output)


        factors_ = tuple(factors+['Nonlinear','Loop 2'])
        ans_ = res[factors_] = measure(nonlinear(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Nonlinear','Loop 3'])
        ans_ = res[factors_] = measure(nonlinear(NTerms, 3), n=N)
        print_results(factors_, ans_, output)

    factors_ = tuple(factors+['Polynomial','Loop 3'])
    ans_ = res[factors_] = measure(polynomial(NTerms, 3), n=N)
    print_results(factors_, ans_, output)


def remap_keys(mapping):
    return [{'factors':k, 'performance': v} for k, v in mapping.items()]

#
# MAIN
#
res = {}

#runall(["COOPR3"], res)

#EXPR.set_expression_tree_format(EXPR.common.Mode.pyomo4_trees) 
#runall(["PYOMO4"], res)

#import pdb; pdb.set_trace()

EXPR.set_expression_tree_format(EXPR.common.Mode.pyomo5_trees) 
runall(["PYOMO5"], res)


print("HERE")

if args.output:
    if args.output.endswith(".csv"):
        #
        # Write csv file
        #
        print("HERE")
        perf_types = sorted(next(iter(res.values())).keys())
        print("HERE")
        res_ = [ list(key) + [res[key][k]['mean'] for k in perf_types] for key in sorted(res.keys())]
        print("HERE")
        with open(args.output, 'w') as OUTPUT:
            import csv
            writer = csv.writer(OUTPUT)
            writer.writerow(['Version', 'ExprType', 'ExprNum'] + perf_types)
            for line in res_:
                writer.writerow(line)

    elif args.output.endswith(".json"):
        res_ = {'script': sys.argv[0], 'NTerms':NTerms, 'NTrials':N, 'data': remap_keys(res), 'pyomo_version':pyomo.version.version, 'pyomo_versioninfo':pyomo.version.version_info[:3]}
        #
        # Write json file
        #
        with open(args.output, 'w') as OUTPUT:
            import json
            json.dump(res_, OUTPUT)

    else:
        print("Unknown output format for file '%s'" % args.output)
