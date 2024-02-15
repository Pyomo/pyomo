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

# This is a file for testing miscellaneous code snippets from the DAE chapter
import pyomo.environ as pyo
import pyomo.dae as dae

m = pyo.ConcreteModel()
m.t = dae.ContinuousSet(bounds=(0, 1))
m.x1 = pyo.Var(m.t)

# @second-deriv:
m.dx1dt2 = dae.DerivativeVar(m.x1, wrt=(m.t, m.t))
# @:second-deriv

# @finite-diff:
discretizer = pyo.TransformationFactory('dae.finite_difference')
discretizer.apply_to(m, nfe=20, wrt=m.t, scheme='BACKWARD')
# @:finite-diff

print('First Model')
print(len(m.t))
print(len(m.x1))

m = pyo.ConcreteModel()
m.t1 = dae.ContinuousSet(bounds=(0, 1))
m.t2 = dae.ContinuousSet(bounds=(0, 1))

m.x1 = pyo.Var(m.t1, m.t2)
m.dx1dt2 = dae.DerivativeVar(m.x1, wrt=(m.t1, m.t2))

# @finite-diff2:
# Apply multiple finite difference schemes
discretizer = pyo.TransformationFactory('dae.finite_difference')
discretizer.apply_to(m, wrt=m.t1, nfe=10, scheme='BACKWARD')
discretizer.apply_to(m, wrt=m.t2, nfe=100, scheme='FORWARD')
# @:finite-diff2

print('Second Model')
print(len(m.t1))
print(len(m.t2))
print(len(m.x1))

m = pyo.ConcreteModel()
m.t1 = dae.ContinuousSet(bounds=(0, 1))
m.t2 = dae.ContinuousSet(bounds=(0, 1))

m.x1 = pyo.Var(m.t1, m.t2)
m.dx1dt2 = dae.DerivativeVar(m.x1, wrt=(m.t1, m.t2))

# @colloc2:
# Apply multiple collocation schemes
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, wrt=m.t1, nfe=4, ncp=6, scheme='LAGRANGE-LEGENDRE')
discretizer.apply_to(m, wrt=m.t2, nfe=10, ncp=3, scheme='LAGRANGE-RADAU')
# @:colloc2

print('Third Model')
print(len(m.t1))
print(len(m.t2))
print(len(m.x1))

m = pyo.ConcreteModel()
m.t1 = dae.ContinuousSet(bounds=(0, 1))
m.t2 = dae.ContinuousSet(bounds=(0, 1))

m.x1 = pyo.Var(m.t1, m.t2)
m.dx1dt2 = dae.DerivativeVar(m.x1, wrt=(m.t1, m.t2))

# @finite-colloc:
# Apply a combination of finite difference and
# collocation schemes
discretizer1 = pyo.TransformationFactory('dae.finite_difference')
discretizer2 = pyo.TransformationFactory('dae.collocation')
discretizer1.apply_to(m, wrt=m.t1, nfe=10, scheme='BACKWARD')
discretizer2.apply_to(m, wrt=m.t2, nfe=5, ncp=3, scheme='LAGRANGE-RADAU')
# @:finite-colloc

print('Fourth Model')
print(len(m.t1))
print(len(m.t2))
print(len(m.x1))
