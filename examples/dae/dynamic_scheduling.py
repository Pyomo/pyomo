#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# This is a toy example for scheduling a sequence of reactions taking
# place in a single reactor. It combines the Pyomo DAE and GDP
# packages and includes modeling concepts from the DAE car example and
# the GDP jobshop example.

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction

m = pyo.ConcreteModel()

m.products = pyo.Set(initialize=['A', 'B'])
m.AthenB = pyo.Set(initialize=[0, 1])

m.tau = ContinuousSet(bounds=[0, 1])  # Unscaled Time

m.k = pyo.Param(m.products, initialize={'A': 1, 'B': 5})  # Reaction Rates

# Cost of having 'A' or 'B' in the final product stream
m.cost = pyo.Param(m.products, initialize={'A': 15000, 'B': 20000})

m.tstart = pyo.Var(m.products, bounds=(0, None))  # Start Time
m.tproc = pyo.Var(m.products, bounds=(0, None))  # Processing Time
m.time = pyo.Var(m.products, m.tau, bounds=(0, None))  # Scaled time over each job
m.totaltime = pyo.Var()  # Total job time

m.c = pyo.Var(m.products, m.tau, bounds=(0, None))
m.dc = DerivativeVar(m.c, wrt=m.tau)
m.dtime = DerivativeVar(m.time, wrt=m.tau)

# Initial concentrations
m.c['A', 0].fix(4)
m.c['B', 0].fix(3)


# Reaction kinetics
def _diffeq(m, p, t):
    return m.dc[p, t] == -m.tproc[p] * m.k[p] * m.c[p, t]


m.diffeq = pyo.Constraint(m.products, m.tau, rule=_diffeq)

# Initial time
m.time['A', 0].fix(0)
m.time['B', 0].fix(0)


# Bound on the final concentration of reactants
def _finalc(m, p):
    return m.c[p, 1] <= 0.001


m.finalc = pyo.Constraint(m.products, rule=_finalc)


# Scaled time
def _diffeqtime(m, p, t):
    return m.dtime[p, t] == m.tproc[p]


m.diffeqtime = pyo.Constraint(m.products, m.tau, rule=_diffeqtime)


# No clash disjuncts
def _noclash(disjunct, AthenB):
    model = disjunct.model()
    if AthenB:
        e = model.tstart['A'] + model.tproc['A'] <= model.tstart['B']
        disjunct.c = pyo.Constraint(expr=e)
    else:
        e = model.tstart['B'] + model.tproc['B'] <= model.tstart['A']
        disjunct.c = pyo.Constraint(expr=e)


m.noclash = Disjunct(m.AthenB, rule=_noclash)


# Define the disjunctions: either job I occurs before K or K before I
def _disj(model):
    return [model.noclash[AthenB] for AthenB in model.AthenB]


m.disj = Disjunction(rule=_disj)


# Due Time
def _duetime(m):
    return m.tstart['B'] + m.tproc['B'] <= 2.0


m.duetime = pyo.Constraint(rule=_duetime)


# Feasibility
def _feas(m, p):
    return m.totaltime >= m.tstart[p] + m.tproc[p]


m.feas = pyo.Constraint(m.products, rule=_feas)


# Objective
def _obj(m):
    return m.totaltime + sum(m.cost[p] * m.c[p, 1] for p in m.products)


m.obj = pyo.Objective(rule=_obj)

# Discretize model
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=5, ncp=3)

# Reformulate Disjuncts
gdp_relax = pyo.TransformationFactory('gdp.bigm')
gdp_relax.apply_to(m, default_bigM=50.0)

# Solve the model
solver = pyo.SolverFactory('couenne')
solver.solve(m, tee=True)

# Plot the results
import matplotlib.pyplot as plt

timeA = [pyo.value(m.time['A', i]) + pyo.value(m.tstart['A']) for i in m.tau]
timeB = [pyo.value(m.time['B', i]) + pyo.value(m.tstart['B']) for i in m.tau]

concA = [pyo.value(m.c['A', i]) for i in m.tau]
concB = [pyo.value(m.c['B', i]) for i in m.tau]

plt.plot(timeA, concA, 'r', label='Reactant A')
plt.plot(timeB, concB, 'b', label='Reactant B')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Concentration in Reactor')
plt.show()
