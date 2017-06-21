#
# This script runs performance tests on expressions
#

import pprint as pp
from pyomo.environ import *
from pyomo.core.base.expr_common import _clear_expression_pool
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

sys.setrecursionlimit(1000000)
#NTerms = 100000
#N = 50
NTerms = 1000
N = 25


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

    gc.collect()
    _clear_expression_pool()
    start = time.time()
    #
    r_ = generate_canonical_repn(expr)
    #
    stop = time.time()
    seconds['generate_canonical'] = stop-start

    return seconds


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

        expr.to_string()
        return seconds

    return f


if True:

    print("Linear Tests: Loop 1")
    sec1 = measure(linear(NTerms, 1), n=N)
    pp.pprint(sec1)

    print("")

    print("Linear Tests: Loop 2")
    sec2 = measure(linear(NTerms, 2), n=N)
    pp.pprint(sec2)

    print("")

    print("Linear Tests: Loop 3")
    sec3 = measure(linear(NTerms, 3), n=N)
    pp.pprint(sec3)

    print("")

    print("Bilinear Tests: Loop 1")
    sec1 = measure(bilinear(NTerms, 1), n=N)
    pp.pprint(sec1)

    print("")

    print("Bilinear Tests: Loop 2")
    sec2 = measure(bilinear(NTerms, 2), n=N)
    pp.pprint(sec2)

    print("")

    print("Bilinear Tests: Loop 3")
    sec3 = measure(bilinear(NTerms, 3), n=N)
    pp.pprint(sec3)

    print("")

    print("Nonlinear Tests: Loop 2")
    sec2 = measure(nonlinear(NTerms, 2), n=N)
    pp.pprint(sec2)

    print("")

    print("Nonlinear Tests: Loop 3")
    sec3 = measure(nonlinear(NTerms, 3), n=N)
    pp.pprint(sec3)

    print("")

print("Polynomial Tests: Loop 3")
sec3 = measure(polynomial(NTerms, 3), n=N)
pp.pprint(sec3)

print("")

