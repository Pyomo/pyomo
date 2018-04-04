#
# This script runs performance tests on expressions
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

## TIMEOUT LOGIC
from functools import wraps
import errno
import os
import signal

try:
    RecursionError
except:
    RecursionError = RuntimeError

coopr3_or_pyomo4 = False
#
# Dummy Sum() function used for Coopr3 tests
#
if coopr3_or_pyomo4:
    def Sum(*args):
        return sum(*args)

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
#NTerms = 100
#N = 1
NTerms = 100000
N = 30 

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
    try:
        seconds['size'] = expr.size()
    except:
        seconds['size'] = -1

    if False:
        #
        # Compression is no longer necessary
        #
        gc.collect()
        _clear_expression_pool()
        try:
            with timeout(seconds=_timeout):
                start = time.time()
                expr = EXPR.compress_expression(expr, verbose=False)
                stop = time.time()
                seconds['compress'] = stop-start
                seconds['compressed_size'] = expr.size()
        except TimeoutError:
            print("TIMEOUT")
            seconds['compressed_size'] = -999.0
        except:
            seconds['compressed_size'] = 0

        # NOTE: All other tests after this are on the compressed expression!

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            expr_ = expr.clone()
            stop = time.time()
            seconds['clone'] = stop-start
    except RecursionError:
        seconds['clone'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['clone'] = -999.0

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            d_ = expr.polynomial_degree()
            stop = time.time()
            seconds['polynomial_degree'] = stop-start
    except RecursionError:
        seconds['polynomial_degree'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['polynomial_degree'] = -999.0

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            s_ = expr.is_constant()
            stop = time.time()
            seconds['is_constant'] = stop-start
    except RecursionError:
        seconds['is_constant'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['is_constant'] = -999.0

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            s_ = expr.is_fixed()
            stop = time.time()
            seconds['is_fixed'] = stop-start
    except RecursionError:
        seconds['is_fixed'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['is_fixed'] = -999.0

    if not coopr3_or_pyomo4:
        gc.collect()
        _clear_expression_pool()
        try:
            from pyomo.repn import generate_standard_repn
            with timeout(seconds=_timeout):
                start = time.time()
                r_ = generate_standard_repn(expr, quadratic=False)
                stop = time.time()
                seconds['generate_repn'] = stop-start
        except RecursionError:
            seconds['generate_repn'] = -888.0
        except ImportError:
            seconds['generate_repn'] = -999.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['generate_repn'] = -999.0

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            s_ = expr.is_constant()
            stop = time.time()
            seconds['is_constant'] = stop-start
    except RecursionError:
        seconds['is_constant'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['is_constant'] = -999.0

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            s_ = expr.is_fixed()
            stop = time.time()
            seconds['is_fixed'] = stop-start
    except RecursionError:
        seconds['is_fixed'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['is_fixed'] = -999.0

    if coopr3_or_pyomo4:
        gc.collect()
        _clear_expression_pool()
        try:
            from pyomo.repn import generate_ampl_repn
            with timeout(seconds=_timeout):
                start = time.time()
                r_ = generate_ampl_repn(expr)
                stop = time.time()
                seconds['generate_repn'] = stop-start
        except RecursionError:
            seconds['generate_repn'] = -888.0
        except ImportError:
            seconds['generate_repn'] = -999.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['generate_repn'] = -999.0

    return seconds

#
# Evaluate standard operations on an expression
#
def evaluate_all(expr, seconds):
    try:
        seconds['size'] = sum(e.size() for e in expr)
    except:
        seconds['size'] = -1

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            for e in expr:
                e.clone()
            stop = time.time()
            seconds['clone'] = stop-start
    except RecursionError:
        seconds['clone'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['clone'] = -999.0

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            for e in expr:
                e.polynomial_degree()
            stop = time.time()
            seconds['polynomial_degree'] = stop-start
    except RecursionError:
        seconds['polynomial_degree'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['polynomial_degree'] = -999.0

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            for e in expr:
                e.is_constant()
            stop = time.time()
            seconds['is_constant'] = stop-start
    except RecursionError:
        seconds['is_constant'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['is_constant'] = -999.0

    gc.collect()
    _clear_expression_pool()
    try:
        with timeout(seconds=_timeout):
            start = time.time()
            for e in expr:
                e.is_fixed()
            stop = time.time()
            seconds['is_fixed'] = stop-start
    except RecursionError:
        seconds['is_fixed'] = -888.0
    except TimeoutError:
        print("TIMEOUT")
        seconds['is_fixed'] = -999.0

    if not coopr3_or_pyomo4:
        gc.collect()
        _clear_expression_pool()
        if True:
            from pyomo.repn import generate_standard_repn
            with timeout(seconds=_timeout):
                start = time.time()
                for e in expr:
                    generate_standard_repn(e, quadratic=False)
                stop = time.time()
                seconds['generate_repn'] = stop-start
        try:
            from pyomo.repn import generate_standard_repn
            with timeout(seconds=_timeout):
                start = time.time()
                for e in expr:
                    generate_standard_repn(e, quadratic=False)
                stop = time.time()
                seconds['generate_repn'] = stop-start
        except RecursionError:
            seconds['generate_repn'] = -888.0
        except ImportError:
            seconds['generate_repn'] = -999.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['generate_repn'] = -999.0

    if coopr3_or_pyomo4:
        gc.collect()
        _clear_expression_pool()
        try:
            from pyomo.repn import generate_ampl_repn
            with timeout(seconds=_timeout):
                start = time.time()
                for e in expr:
                    generate_ampl_repn(e)
                stop = time.time()
                seconds['generate_repn'] = stop-start
        except RecursionError:
            seconds['generate_repn'] = -888.0
        except ImportError:
            seconds['generate_repn'] = -999.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['generate_repn'] = -999.0

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
        try:
            with EXPR.clone_counter as ctr:
                nclones = ctr.count

                with timeout(seconds=_timeout):
                    #
                    if flag == 1:
                        start = time.time()
                        expr = sum_product(model.p, model.x)
                        stop = time.time()
                    elif flag == 2:
                        start = time.time()
                        expr=sum(model.p[i]*model.x[i] for i in model.A)
                        stop = time.time()
                    elif flag == 3:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr += model.p[i] * model.x[i]
                        stop = time.time()
                    elif flag == 4:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr = expr + model.p[i] * model.x[i]
                        stop = time.time()
                    elif flag == 5:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr = model.p[i] * model.x[i] + expr
                        stop = time.time()
                    elif flag == 6:
                        start = time.time()
                        expr=Sum(model.p[i]*model.x[i] for i in model.A)
                        stop = time.time()
                    elif flag == 7:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr += model.p[i] * (1 + model.x[i])
                        stop = time.time()
                    elif flag == 8:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr += (model.x[i]+model.x[i])
                        stop = time.time()
                    elif flag == 9:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr += model.p[i]*(model.x[i]+model.x[i])
                        stop = time.time()
                    elif flag == 12:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            expr=sum((model.p[i]*model.x[i] for i in model.A), expr)
                        stop = time.time()
                    elif flag == 13:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr += model.p[i] * model.x[i]
                        stop = time.time()
                    elif flag == 14:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr = expr + model.p[i] * model.x[i]
                        stop = time.time()
                    elif flag == 15:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr = model.p[i] * model.x[i] + expr
                        stop = time.time()
                    elif flag == 17:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr += model.p[i] * (1 + model.x[i])
                        stop = time.time()
                    #
                    seconds['construction'] = stop-start
                    seconds['nclones'] = ctr.count - nclones
                seconds = evaluate(expr, seconds)
        except RecursionError:
            seconds['construction'] = -888.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['construction'] = -999.0
        return seconds

    return f

#
# Create a linear expression
#
def simple_linear(N, flag):

    def f():
        seconds = {}

        model = ConcreteModel()
        model.A = RangeSet(N)
        model.p = Param(model.A, default=2)
        model.x = Var(model.A, initialize=2)

        gc.collect()
        _clear_expression_pool()
        try:
            with EXPR.clone_counter as ctr:
                nclones = ctr.count

                with timeout(seconds=_timeout):
                    #
                    if flag == 1:
                        start = time.time()
                        expr = sum_product(model.p, model.x)
                        stop = time.time()
                    elif flag == 2:
                        start = time.time()
                        expr=sum(model.p[i]*model.x[i] for i in model.A)
                        stop = time.time()
                    elif flag == 3:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr += model.p[i] * model.x[i]
                        stop = time.time()
                    elif flag == 4:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr = expr + model.p[i] * model.x[i]
                        stop = time.time()
                    elif flag == 5:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr = model.p[i] * model.x[i] + expr
                        stop = time.time()
                    elif flag == 6:
                        start = time.time()
                        expr=Sum(model.p[i]*model.x[i] for i in model.A)
                        stop = time.time()
                    elif flag == 7:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr += model.p[i] * (1 + model.x[i])
                        stop = time.time()
                    elif flag == 8:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr += (model.x[i]+model.x[i])
                        stop = time.time()
                    elif flag == 9:
                        start = time.time()
                        expr=0
                        for i in model.A:
                            expr += model.p[i]*(model.x[i]+model.x[i])
                        stop = time.time()
                    elif flag == 12:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            expr=sum((model.p[i]*model.x[i] for i in model.A), expr)
                        stop = time.time()
                    elif flag == 13:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr += model.p[i] * model.x[i]
                        stop = time.time()
                    elif flag == 14:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr = expr + model.p[i] * model.x[i]
                        stop = time.time()
                    elif flag == 15:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr = model.p[i] * model.x[i] + expr
                        stop = time.time()
                    elif flag == 17:
                        start = time.time()
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr += model.p[i] * (1 + model.x[i])
                        stop = time.time()
                    #
                    seconds['construction'] = stop-start
                    seconds['nclones'] = ctr.count - nclones
                seconds = evaluate(expr, seconds)
        except RecursionError:
            seconds['construction'] = -888.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['construction'] = -999.0
        return seconds

    return f

#
# Create a nested linear expression
#
def nested_linear(N, flag):

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
        try:
            with EXPR.clone_counter as ctr:
                nclones = ctr.count
                with timeout(seconds=_timeout):
                    start = time.time()
                    #
                    if flag == 1:
                        expr = 2* sum_product(model.p, model.x)
                    elif flag == 2:
                        expr= 2 * sum(model.p[i]*model.x[i] for i in model.A)
                    elif flag == 3:
                        expr=0
                        for i in model.A:
                            expr += model.p[i] * model.x[i]
                        expr *= 2
                    elif flag == 4:
                        expr=0
                        for i in model.A:
                            expr = expr + model.p[i] * model.x[i]
                        expr *= 2
                    elif flag == 5:
                        expr=0
                        for i in model.A:
                            expr = model.p[i] * model.x[i] + expr
                        expr *= 2
                    elif flag == 6:
                        expr= 2 * Sum(model.p[i]*model.x[i] for i in model.A)
                    elif flag == 12:
                        with EXPR.linear_expression as expr:
                            expr= 2 * sum((model.p[i]*model.x[i] for i in model.A), expr)
                    elif flag == 13:
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr += model.p[i] * model.x[i]
                            expr *= 2
                    elif flag == 14:
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr = expr + model.p[i] * model.x[i]
                            expr *= 2
                    elif flag == 15:
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr = model.p[i] * model.x[i] + expr
                            expr *= 2
                    #
                    stop = time.time()
                    seconds['construction'] = stop-start
                    seconds['nclones'] = ctr.count - nclones
                seconds = evaluate(expr, seconds)
        except RecursionError:
            seconds['construction'] = -888.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['construction'] = -999.0
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
        model.q = Param(model.A, initialize=2, mutable=True)

        gc.collect()
        _clear_expression_pool()
        try:
            with EXPR.clone_counter as ctr:
                nclones = ctr.count

                with timeout(seconds=_timeout):
                    start = time.time()
                    #
                    if flag == 1:
                        expr = sum_product(model.p, model.q, index=model.A)
                    elif flag == 2:
                        expr=sum(model.p[i]*model.q[i] for i in model.A)
                    elif flag == 3:
                        expr=0
                        for i in model.A:
                            expr += model.p[i] * model.q[i]
                    elif flag == 4:
                        expr=Sum(model.p[i]*model.q[i] for i in model.A)
                    elif flag == 12:
                        with EXPR.linear_expression as expr:
                            expr=sum((model.p[i]*model.q[i] for i in model.A), expr)
                    elif flag == 13:
                        with EXPR.linear_expression as expr:
                            for i in model.A:
                                expr += model.p[i] * model.q[i]
                    #
                    stop = time.time()
                    seconds['construction'] = stop-start
                    seconds['nclones'] = ctr.count - nclones
                seconds = evaluate(expr, seconds)
        except RecursionError:
            seconds['construction'] = -888.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['construction'] = -999.0
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
        try:
            with EXPR.clone_counter as ctr:
                nclones = ctr.count

                with timeout(seconds=_timeout):
                    start = time.time()
                    #
                    if flag == 1:
                        expr = sum_product(model.p, model.x, model.y)
                    elif flag == 2:
                        expr=sum(model.p[i]*model.x[i]*model.y[i] for i in model.A)
                    elif flag == 3:
                        expr=0
                        for i in model.A:
                            expr += model.p[i] * model.x[i] * model.y[i]
                    elif flag == 4:
                        expr=Sum(model.p[i]*model.x[i]*model.y[i] for i in model.A)
                    elif flag == 12:
                        with EXPR.quadratic_expression as expr:
                            expr=sum((model.p[i]*model.x[i]*model.y[i] for i in model.A), expr)
                    elif flag == 13:
                        with EXPR.quadratic_expression as expr:
                            for i in model.A:
                                expr += model.p[i] * model.x[i] * model.y[i]
                    #
                    stop = time.time()
                    seconds['construction'] = stop-start
                    seconds['nclones'] = ctr.count - nclones
                seconds = evaluate(expr, seconds)
        except RecursionError:
            seconds['construction'] = -888.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['construction'] = -999.0
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
        try:
            with EXPR.clone_counter as ctr:
                nclones = ctr.count

                with timeout(seconds=_timeout):
                    start = time.time()
                    #
                    if flag == 2:
                        expr=sum(model.p[i]*tan(model.x[i]) for i in model.A)
                    elif flag == 3:
                        expr=0
                        for i in model.A:
                            expr += model.p[i] * tan(model.x[i])
                    elif flag == 4:
                        expr=Sum(model.p[i]*tan(model.x[i]) for i in model.A)
                    if flag == 12:
                        with EXPR.nonlinear_expression as expr:
                            expr=sum((model.p[i]*tan(model.x[i]) for i in model.A), expr)
                    elif flag == 13:
                        with EXPR.nonlinear_expression as expr:
                            for i in model.A:
                                expr += model.p[i] * tan(model.x[i])
                    #
                    stop = time.time()
                    seconds['construction'] = stop-start
                    seconds['nclones'] = ctr.count - nclones
                seconds = evaluate(expr, seconds)
        except RecursionError:
            seconds['construction'] = -888.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['construction'] = -999.0
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

        gc.collect()
        _clear_expression_pool()
        try:
            with EXPR.clone_counter as ctr:
                nclones = ctr.count

                with timeout(seconds=_timeout):
                    start = time.time()
                    #
                    if True:
                        expr=0
                        for i in model.A:
                            expr = model.x[i] * (1 + expr)
                    #
                    stop = time.time()
                    seconds['construction'] = stop-start
                    seconds['nclones'] = ctr.count - nclones
                seconds = evaluate(expr, seconds)
        except RecursionError:
            seconds['construction'] = -888.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['construction'] = -999.0
        return seconds

    return f


#
# Create an expression that is a large product
#
def product(N, flag):

    def f():
        seconds = {}

        model = ConcreteModel()
        model.A = RangeSet(N)
        model.p = Param(model.A, default=2)
        model.x = Var(initialize=2)

        gc.collect()
        _clear_expression_pool()
        try:
            with EXPR.clone_counter as ctr:
                nclones = ctr.count

                with timeout(seconds=_timeout):
                    start = time.time()
                    #
                    if flag == 1:
                        expr=model.x+model.x
                        for i in model.A:
                            expr = model.p[i]*expr
                    elif flag == 2:
                        expr=model.x+model.x
                        for i in model.A:
                            expr *= model.p[i]
                    elif flag == 3:
                        expr=(model.x+model.x) * prod(model.p[i] for i in model.A)
                    #
                    stop = time.time()
                    seconds['construction'] = stop-start
                    seconds['nclones'] = ctr.count - nclones
                seconds = evaluate(expr, seconds)
        except RecursionError:
            seconds['construction'] = -888.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['construction'] = -999.0
        return seconds

    return f


#
# Create many small linear expressions
#
def many_linear(N, flag):

    def f():
        seconds = {}

        model = ConcreteModel()
        model.A = RangeSet(N)
        model.p = Param(model.A, default=2)
        model.x = Var(model.A, initialize=2)

        gc.collect()
        _clear_expression_pool()
        if True:
            with EXPR.clone_counter as ctr:
                nclones = ctr.count

                with timeout(seconds=_timeout):
                    start = time.time()
                    #
                    expr = []
                    if flag == 2:
                        for i in model.A:
                            expr.append( model.x[1] + model.x[i] )
                    #
                    stop = time.time()
                    seconds['construction'] = stop-start
                    seconds['nclones'] = ctr.count - nclones
                seconds = evaluate_all(expr, seconds)
        try:
            pass
        except RecursionError:
            seconds['construction'] = -888.0
        except TimeoutError:
            print("TIMEOUT")
            seconds['construction'] = -999.0
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

    if True:
        factors_ = tuple(factors+['ManyLinear','Loop 2'])
        ans_ = res[factors_] = measure(many_linear(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

    if True:
        factors_ = tuple(factors+['Constant','Loop 1'])
        ans_ = res[factors_] = measure(constant(NTerms, 1), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Constant','Loop 2'])
        ans_ = res[factors_] = measure(constant(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Constant','Loop 12'])
        ans_ = res[factors_] = measure(constant(NTerms, 12), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Constant','Loop 3'])
        ans_ = res[factors_] = measure(constant(NTerms, 3), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Constant','Loop 13'])
        ans_ = res[factors_] = measure(constant(NTerms, 13), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Constant','Loop 4'])
        ans_ = res[factors_] = measure(constant(NTerms, 4), n=N)
        print_results(factors_, ans_, output)

    if True:
        factors_ = tuple(factors+['Linear','Loop 1'])
        ans_ = res[factors_] = measure(linear(NTerms, 1), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 2'])
        ans_ = res[factors_] = measure(linear(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 12'])
        ans_ = res[factors_] = measure(linear(NTerms, 12), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 3'])
        ans_ = res[factors_] = measure(linear(NTerms, 3), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 13'])
        ans_ = res[factors_] = measure(linear(NTerms, 13), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 4'])
        ans_ = res[factors_] = measure(linear(NTerms, 4), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 14'])
        ans_ = res[factors_] = measure(linear(NTerms, 14), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 5'])
        ans_ = res[factors_] = measure(linear(NTerms, 5), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 15'])
        ans_ = res[factors_] = measure(linear(NTerms, 15), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 6'])
        ans_ = res[factors_] = measure(linear(NTerms, 6), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 7'])
        ans_ = res[factors_] = measure(linear(NTerms, 7), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 17'])
        ans_ = res[factors_] = measure(linear(NTerms, 17), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 8'])
        ans_ = res[factors_] = measure(linear(NTerms, 8), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Linear','Loop 9'])
        ans_ = res[factors_] = measure(linear(NTerms, 9), n=N)
        print_results(factors_, ans_, output)

    if True:
        factors_ = tuple(factors+['SimpleLinear','Loop 1'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 1), n=N)
        print_results(factors_, ans_, output)

    if True:
        factors_ = tuple(factors+['SimpleLinear','Loop 2'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 12'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 12), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 3'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 3), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 13'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 13), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 4'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 4), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 14'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 14), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 5'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 5), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 15'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 15), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 7'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 7), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 17'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 17), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 8'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 8), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['SimpleLinear','Loop 9'])
        ans_ = res[factors_] = measure(simple_linear(NTerms, 9), n=N)
        print_results(factors_, ans_, output)


    if True:
        factors_ = tuple(factors+['NestedLinear','Loop 1'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 1), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['NestedLinear','Loop 2'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['NestedLinear','Loop 12'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 12), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['NestedLinear','Loop 3'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 3), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['NestedLinear','Loop 13'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 13), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['NestedLinear','Loop 4'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 4), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['NestedLinear','Loop 14'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 14), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['NestedLinear','Loop 5'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 5), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['NestedLinear','Loop 15'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 15), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['NestedLinear','Loop 6'])
        ans_ = res[factors_] = measure(nested_linear(NTerms, 6), n=N)
        print_results(factors_, ans_, output)


    if True:
        factors_ = tuple(factors+['Bilinear','Loop 1'])
        ans_ = res[factors_] = measure(bilinear(NTerms, 1), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Bilinear','Loop 2'])
        ans_ = res[factors_] = measure(bilinear(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Bilinear','Loop 12'])
        ans_ = res[factors_] = measure(bilinear(NTerms, 12), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Bilinear','Loop 3'])
        ans_ = res[factors_] = measure(bilinear(NTerms, 3), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Bilinear','Loop 13'])
        ans_ = res[factors_] = measure(bilinear(NTerms, 13), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Bilinear','Loop 4'])
        ans_ = res[factors_] = measure(bilinear(NTerms, 4), n=N)
        print_results(factors_, ans_, output)


    if True:
        factors_ = tuple(factors+['Nonlinear','Loop 2'])
        ans_ = res[factors_] = measure(nonlinear(NTerms, 2), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Nonlinear','Loop 12'])
        ans_ = res[factors_] = measure(nonlinear(NTerms, 12), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Nonlinear','Loop 3'])
        ans_ = res[factors_] = measure(nonlinear(NTerms, 3), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Nonlinear','Loop 13'])
        ans_ = res[factors_] = measure(nonlinear(NTerms, 13), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Nonlinear','Loop 4'])
        ans_ = res[factors_] = measure(nonlinear(NTerms, 4), n=N)
        print_results(factors_, ans_, output)


    if True:
        factors_ = tuple(factors+['Polynomial','Loop 3'])
        ans_ = res[factors_] = measure(polynomial(NTerms, 3), n=N)
        print_results(factors_, ans_, output)


    if True:
        factors_ = tuple(factors+['Product','Loop 1'])
        ans_ = res[factors_] = measure(polynomial(NTerms, 1), n=N)
        print_results(factors_, ans_, output)

        factors_ = tuple(factors+['Product','Loop 2'])
        ans_ = res[factors_] = measure(polynomial(NTerms, 2), n=N)
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

#EXPR.set_expression_tree_format(EXPR.common.Mode.pyomo5_trees) 
#import cProfile
#cProfile.run('runall(["PYOMO5"], res)', 'restats4')
runall(["PYOMO5"], res)

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
        res_ = {'script': sys.argv[0], 'NTerms':NTerms, 'NTrials':N, 'data': remap_keys(res), 'pyomo_version':pyomo.version.version, 'pyomo_versioninfo':pyomo.version.version_info[:3]}
        #
        # Write json file
        #
        with open(args.output, 'w') as OUTPUT:
            import json
            json.dump(res_, OUTPUT)

    else:
        print("Unknown output format for file '%s'" % args.output)
