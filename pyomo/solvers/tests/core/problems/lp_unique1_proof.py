# This is a computational proof that lp_unique1.py has a unique primal and dual solution.

from pyomo.core import *
from pyomo.opt import SolverFactory

def get_model(flag, fixprim, fixdual):
    model = ConcreteModel()

    model.obj = Param(default=20)

    model.n = Param(default=7)
    model.m = Param(default=7)

    model.N = RangeSet(1,model.n)
    model.M = RangeSet(1,model.m)

    def c_rule(model, j):
        return 5 if j<5 else 9.0/2
    model.c = Param(model.N)

    def b_rule(model, i):
        if i == 4:
            i = 5
        elif i == 5:
            i = 4
        return 5 if i<5 else 7.0/2
    model.b = Param(model.M)

    def A_rule(model, i, j):
        if i == 4:
            i = 5
        elif i == 5:
            i = 4
        return 2 if i==j else 1
    model.A = Param(model.M, model.N)

    model.x = Var(model.N, within=NonNegativeReals)
    model.y = Var(model.M, within=NonNegativeReals)
    model.xx = Var([1,2], model.N, within=NonNegativeReals)
    model.yy = Var([1,2], model.M, within=NonNegativeReals)

    if flag:
        model.ydiff = Objective(expr=model.yy[2,fixdual] - model.yy[1,fixdual])
        
        def yext_rule(model, k):
            return sum(model.b[i]*model.yy[k,i] for i in model.M) == model.obj
        model.yext = Constraint([1,2])

        def dualcons_rule(model, k, j):
            return sum(model.A[i,j]*model.yy[k,i] for i in model.N) <= model.c[j]
        model.dualcons = Constraint([1,2], model.N)
    else:
        model.xdiff = Objective(expr=model.xx[2,fixprim] - model.xx[1,fixprim])

        def xext_rule(model, k):
            return sum(model.c[j]*model.xx[k,j] for j in model.N) == model.obj
        model.xext = Constraint([1,2])

        def primcons_rule(model, k, i):
            return sum(model.A[i,j]*model.xx[k,j] for j in model.M) >= model.b[i]
        model.primcons = Constraint([1,2], model.M)

    model.create()
    return model


#
# We iterate over all primal and dual indices, fixing each in turn.  Then, we solve a max-difference
# LP problem and verify that the difference is zero.  This demonstrates that there is no alternative 
# primal/dual solution, given that fixed value.
#
opt = SolverFactory('glpk')

for i in range(1,8):
    model = get_model(True, i, 1)
    results = opt.solve(model)
    if results.solution[0].objective['ydiff'].value != 0:
        print "ERROR: nonzero difference i=%d ydiff=%f" % (i, results.solution[0].objective['ydiff'].value)

for i in range(1,8):
    model = get_model(False, 1, i)
    results = opt.solve(model)
    if results.solution[0].objective['xdiff'].value != 0:
        print "ERROR: nonzero difference i=%d xdiff=%f" % (i, results.solution[0].objective['xdiff'].value)

print "SUCCESS!"
