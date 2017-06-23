#
# This script runs performance tests on expressions
#

import pprint as pp
from pyomo.environ import *
from pyomo.core.base.expr_common import _clear_expression_pool
from pyomo.core.base import expr 
from pyomo.repn import generate_canonical_repn
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

sys.setrecursionlimit(1000000)
#NTerms = 100000
#N = 50
NTerms = 1000
N = 25


try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "h:", ["help", "output=", 'num=', 'terms='])
except getopt.GetoptError as err:
    # print help information and exit:
    print(str(err))  # will print something like "option -a not recognized"
    print(sys.argv[0] + " -h --num=<ntrials> --terms=<nterms> --output=<filename>")
    sys.exit(2)

ofile = None
for o, a in opts:
    if o in ("-h", "--help"):
        print(sys.argv[0] + " -h --num=<ntrials> --terms=<nterms> --output=<filename>")
        sys.exit()
    elif o == "--output":
        ofile = a
    elif o == "--num":
        N = int(a)
    elif o == "--terms":
        NTerms = int(a)
    else:
        assert False, "unhandled option"

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
# Evaluate standard operations on an expression
#
def evaluate(expr, seconds):
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

runall(["COOPR3"], res)

expr.set_expression_tree_format(expr.common.Mode.pyomo4_trees) 
runall(["PYOMO4"], res)


if ofile:
    import json
    OUTPUT = open(ofile, 'w')
    res_ = {'script': sys.argv[0], 'NTerms':NTerms, 'NTrials':N, 'data': remap_keys(res)}
    json.dump(res_, OUTPUT)
    OUTPUT.close()

