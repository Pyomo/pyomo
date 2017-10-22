from pyomo.environ import *
import pyomo.core.kernel.expr as EXPR
import timeit
import signal

coopr3 = False
pyomo4 = False

if coopr3 or pyomo4:
    from pyomo.repn import generate_ampl_repn
else:
    from pyomo.repn import generate_standard_repn

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


N = 100000

model = ConcreteModel()
model.A = RangeSet(N)
model.p = Param(model.A, default=2)
model.x = Var(model.A, initialize=2)


def linear(flag):
    if flag == 1:
        expr = summation(model.p, model.x)
    elif flag == 6:
        expr=Sum(model.p[i]*model.x[i] for i in model.A)

    elif flag == 2:
        expr=sum(model.p[i]*model.x[i] for i in model.A)
    elif flag == 3:
        expr=0
        for i in model.A:
            expr += model.p[i] * model.x[i]
    elif flag == 4:
        try:
            with timeout(10):
                expr=0
                for i in model.A:
                    expr = expr + model.p[i] * model.x[i]
        except:
            expr = model.x[1]       # BOGUS
    elif flag == 5:
        try:
            with timeout(10):
                expr=0
                for i in model.A:
                    expr = model.p[i] * model.x[i] + expr
        except:
            expr = model.x[1]       # BOGUS

    elif flag == 12:
        with EXPR.linear_expression as expr:
            expr=sum((model.p[i]*model.x[i] for i in model.A), expr)
    elif flag == 22:
        with EXPR.nonlinear_expression as expr:
            expr=sum((model.p[i]*model.x[i] for i in model.A), expr)
    elif flag == 13:
        with EXPR.linear_expression as expr:
            for i in model.A:
                expr += model.p[i] * model.x[i]
    elif flag == 14:
        with EXPR.linear_expression as expr:
            for i in model.A:
                expr = expr + model.p[i] * model.x[i]
    elif flag == 15:
        with EXPR.linear_expression as expr:
            for i in model.A:
                expr = model.p[i] * model.x[i] + expr

    elif flag == 7:
        expr=0
        for i in model.A:
            expr += model.p[i] * (1 + model.x[i])

    elif flag == 17:
        with EXPR.linear_expression as expr:
            for i in model.A:
                expr += model.p[i] * (1 + model.x[i])

    if coopr3 or pyomo4:
        generate_ampl_repn(expr)
    else:
        generate_standard_repn(EXPR.compress_expression(expr), quadratic=False)

if coopr3:
    import pyomo.core.kernel.expr_coopr3 as COOPR3
    print("REFCOUNT: "+str(COOPR3._getrefcount_available))
    for i in (2,3,7):
        print((i,timeit.timeit('linear(%d)' % i, "from __main__ import linear", number=1)))

if pyomo4:
    import pyomo.core.kernel.expr_pyomo4 as PYOMO4
    EXPR.set_expression_tree_format(EXPR.common.Mode.pyomo4_trees)
    print("REFCOUNT: "+str(PYOMO4._getrefcount_available))
    for i in (2,3,7):
        print((i,timeit.timeit('linear(%d)' % i, "from __main__ import linear", number=1)))

if not (coopr3 or pyomo4):
    import pyomo.core.kernel.expr_pyomo5 as PYOMO5
    print("REFCOUNT: "+str(PYOMO5._getrefcount_available))
    for i in (2,12,22,3,13,4,14,5,15,7,17):
        print((i,timeit.timeit('linear(%d)' % i, "from __main__ import linear", number=1)))

