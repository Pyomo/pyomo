#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

model = pyo.ConcreteModel()

# @initscalarvar:
model.x = pyo.Var(initialize=3.14)
# @:initscalarvar

# for testing
print('3.14 =', pyo.value(model.x))  # 3.14

model = None
model = pyo.ConcreteModel()

# @dictruleinit:
model.A = pyo.Set(initialize=[1, 2, 3])
model.x = pyo.Var(model.A, initialize=3.14)
model.y = pyo.Var(model.A, initialize={1: 1.5, 2: 4.5, 3: 5.5})


def z_init_rule(m, i):
    return float(i) + 0.5


model.z = pyo.Var(model.A, initialize=z_init_rule)
# @:dictruleinit

# for testing
print('[3.14, 3.14, 3.14] =', [pyo.value(model.x[i]) for i in model.x])
print('[1.5, 4.5, 5.5] =', [pyo.value(model.y[i]) for i in model.y])
print('[1.5, 2.5, 3.5] =', [pyo.value(model.z[i]) for i in model.x])

model = None
model = pyo.ConcreteModel()

# @declscalar:
model.x = pyo.Var()
# @:declscalar

# for testing
print("x =", model.x)

model = None
model = pyo.ConcreteModel()

# @domaindecl:
model.A = pyo.Set(initialize=[1, 2, 3])
model.y = pyo.Var(within=model.A)
model.r = pyo.Var(domain=pyo.Reals)
model.w = pyo.Var(within=pyo.Boolean)
# @:domaindecl

# for testing
model.A.pprint()
print(model.y.domain)
print(model.r.domain)
print(model.w.domain)

model = None
model = pyo.ConcreteModel()

# @domaindeclrule:
model.A = pyo.Set(initialize=[1, 2, 3])


def s_domain(model, i):
    return pyo.RangeSet(i, i + 1, 1)  # (start, end, step)


model.s = pyo.Var(model.A, domain=s_domain)
# @:domaindeclrule

# for testing
model.A.pprint()
model.s.pprint()

model = None
model = pyo.ConcreteModel()

# @declbounds:
model.A = pyo.Set(initialize=[1, 2, 3])
model.a = pyo.Var(bounds=(0.0, None))

lower = {1: 2.5, 2: 4.5, 3: 6.5}
upper = {1: 3.5, 2: 4.5, 3: 7.5}


def f(model, i):
    return (lower[i], upper[i])


model.b = pyo.Var(model.A, bounds=f)
# @:declbounds

# for testing
model.A.pprint()
print(model.a.lb)
print(model.a.ub)
print(model.b[3].lb)
print(model.b[3].ub)

model = None
model = pyo.ConcreteModel()

# @declinit:
model.A = pyo.Set(initialize=[1, 2, 3])
model.za = pyo.Var(initialize=9.5, within=pyo.NonNegativeReals)
model.zb = pyo.Var(model.A, initialize={1: 1.5, 2: 4.5, 3: 5.5})
model.zc = pyo.Var(model.A, initialize=2.1)

print(pyo.value(model.za))  # 9.5
print(pyo.value(model.zb[3]))  # 5.5
print(pyo.value(model.zc[3]))  # 2.1
# @:declinit

model = None
model = pyo.ConcreteModel()

# @declinitrule:
model.A = pyo.Set(initialize=[1, 2, 3])


def g(model, i):
    return 3 * i


model.m = pyo.Var(model.A, initialize=g)

print(pyo.value(model.m[1]))  # 3
print(pyo.value(model.m[3]))  # 9
# @:declinitrule

model = None
model = pyo.ConcreteModel()

print("varattrib")
# @varattribdecl:
model.A = pyo.Set(initialize=[1, 2, 3])
model.za = pyo.Var(initialize=9.5, within=pyo.NonNegativeReals)
model.zb = pyo.Var(model.A, initialize={1: 1.5, 2: 4.5, 3: 5.5})
model.zc = pyo.Var(model.A, initialize=2.1)
# @:varattribdecl

# @varattribvaluebounds:
print(pyo.value(model.zb[2]))  # 4.5
print(model.za.lb)  # 0
print(model.za.ub)  # None
# @:varattribvaluebounds

# @varassign:
model.za = 8.5
model.zb[2] = 7.5
# @:varassign

# @varfixed:
model.zb.fix(3.0)
print(model.zb[1].fixed)  # True
print(model.zb[2].fixed)  # True
model.zc[2].fix(3.0)
print(model.zc[1].fixed)  # False
print(model.zc[2].fixed)  # True
# @:varfixed
