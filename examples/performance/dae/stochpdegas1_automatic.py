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

# stochastic pde model for natural gas network
# victor m. zavala / 2013

#

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

model = pyo.AbstractModel()

# sets
model.TF = pyo.Param(within=pyo.NonNegativeReals)


def _tinit(m):
    return [0.5, pyo.value(m.TF)]
    # What it should be to match description in paper
    # return [0,pyo.value(m.TF)]


model.TIME = ContinuousSet(initialize=_tinit)
model.DIS = ContinuousSet(bounds=(0.0, 1.0))
model.S = pyo.Param(within=pyo.PositiveIntegers)
model.SCEN = pyo.RangeSet(1, model.S)

# links
model.LINK = pyo.Set()
model.lstartloc = pyo.Param(model.LINK)
model.lendloc = pyo.Param(model.LINK)
model.ldiam = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
model.llength = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
model.ltype = pyo.Param(model.LINK)


def link_a_init_rule(m):
    return (l for l in m.LINK if m.ltype[l] == "a")


model.LINK_A = pyo.Set(initialize=link_a_init_rule)


def link_p_init_rule(m):
    return (l for l in m.LINK if m.ltype[l] == "p")


model.LINK_P = pyo.Set(initialize=link_p_init_rule)

# nodes
model.NODE = pyo.Set()
model.pmin = pyo.Param(model.NODE, within=pyo.PositiveReals, mutable=True)
model.pmax = pyo.Param(model.NODE, within=pyo.PositiveReals, mutable=True)

# supply
model.SUP = pyo.Set()
model.sloc = pyo.Param(model.SUP)
model.smin = pyo.Param(model.SUP, within=pyo.NonNegativeReals, mutable=True)
model.smax = pyo.Param(model.SUP, within=pyo.NonNegativeReals, mutable=True)
model.scost = pyo.Param(model.SUP, within=pyo.NonNegativeReals)

# demand
model.DEM = pyo.Set()
model.dloc = pyo.Param(model.DEM)
model.d = pyo.Param(model.DEM, within=pyo.PositiveReals, mutable=True)

# physical data
model.eps = pyo.Param(initialize=0.025, within=pyo.PositiveReals)
model.z = pyo.Param(initialize=0.80, within=pyo.PositiveReals)
model.rhon = pyo.Param(initialize=0.72, within=pyo.PositiveReals)
model.R = pyo.Param(initialize=8314.0, within=pyo.PositiveReals)
model.M = pyo.Param(initialize=18.0, within=pyo.PositiveReals)
model.pi = pyo.Param(initialize=3.14, within=pyo.PositiveReals)
model.nu2 = pyo.Param(within=pyo.PositiveReals, mutable=True)
model.lam = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
model.A = pyo.Param(model.LINK, within=pyo.NonNegativeReals, mutable=True)
model.Tgas = pyo.Param(initialize=293.15, within=pyo.PositiveReals)
model.Cp = pyo.Param(initialize=2.34, within=pyo.PositiveReals)
model.Cv = pyo.Param(initialize=1.85, within=pyo.PositiveReals)
model.gam = pyo.Param(initialize=model.Cp / model.Cv, within=pyo.PositiveReals)
model.om = pyo.Param(initialize=(model.gam - 1.0) / model.gam, within=pyo.PositiveReals)

# scaling and constants
model.ffac = pyo.Param(
    within=pyo.PositiveReals, initialize=(1.0e6 * model.rhon) / (24.0 * 3600.0)
)
model.ffac2 = pyo.Param(
    within=pyo.PositiveReals, initialize=(3600.0) / (1.0e4 * model.rhon)
)
model.pfac = pyo.Param(within=pyo.PositiveReals, initialize=1.0e5)
model.pfac2 = pyo.Param(within=pyo.PositiveReals, initialize=1.0e-5)
model.dfac = pyo.Param(within=pyo.PositiveReals, initialize=1.0e-3)
model.lfac = pyo.Param(within=pyo.PositiveReals, initialize=1.0e3)

model.c1 = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
model.c2 = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
model.c3 = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
model.c4 = pyo.Param(within=pyo.PositiveReals, mutable=True)

# cost factors
model.ce = pyo.Param(initialize=0.1, within=pyo.NonNegativeReals)
model.cd = pyo.Param(initialize=1.0e6, within=pyo.NonNegativeReals)
model.cT = pyo.Param(initialize=1.0e6, within=pyo.NonNegativeReals)
model.cs = pyo.Param(initialize=0.0, within=pyo.NonNegativeReals)
model.TDEC = pyo.Param(within=pyo.PositiveReals)

# define stochastic info
model.rand_d = pyo.Param(
    model.SCEN, model.DEM, within=pyo.NonNegativeReals, mutable=True
)


# convert units for input data
def rescale_rule(m):
    for i in m.LINK:
        m.ldiam[i] = m.ldiam[i] * m.dfac
        m.llength[i] = m.llength[i] * m.lfac
        # m.dx[i] = m.llength[i]/float(m.DIS.last())

    for i in m.SUP:
        m.smin[i] = (
            m.smin[i] * m.ffac * m.ffac2
        )  # from scmx106/day to kg/s and then to scmx10-4/hr
        m.smax[i] = (
            m.smax[i] * m.ffac * m.ffac2
        )  # from scmx106/day to kg/s and then to scmx10-4/hr

    for i in m.DEM:
        m.d[i] = m.d[i] * m.ffac * m.ffac2

    for i in m.NODE:
        m.pmin[i] = m.pmin[i] * m.pfac * m.pfac2  # from bar to Pascals and then to bar
        m.pmax[i] = m.pmax[i] * m.pfac * m.pfac2  # from bar to Pascals and then to bar


model.rescale = pyo.BuildAction(rule=rescale_rule)


def compute_constants(m):
    for i in m.LINK:
        m.lam[i] = (2.0 * pyo.log10(3.7 * m.ldiam[i] / (m.eps * m.dfac))) ** (-2.0)
        m.A[i] = (1.0 / 4.0) * m.pi * m.ldiam[i] * m.ldiam[i]
        m.nu2 = m.gam * m.z * m.R * m.Tgas / m.M
        m.c1[i] = (m.pfac2 / m.ffac2) * (m.nu2 / m.A[i])
        m.c2[i] = m.A[i] * (m.ffac2 / m.pfac2)
        m.c3[i] = (
            m.A[i]
            * (m.pfac2 / m.ffac2)
            * (8.0 * m.lam[i] * m.nu2)
            / (m.pi * m.pi * (m.ldiam[i] ** 5.0))
        )
        m.c4 = (1 / m.ffac2) * (m.Cp * m.Tgas)


model.compute_constants = pyo.BuildAction(rule=compute_constants)


# set stochastic demands
def compute_demands_rule(m):
    for k in m.SCEN:
        for j in m.DEM:
            if k == 2:
                m.rand_d[k, j] = 1.1 * m.d[j]
            elif k == 1:
                m.rand_d[k, j] = 1.2 * m.d[j]
            else:
                m.rand_d[k, j] = 1.3 * m.d[j]


model.compute_demands = pyo.BuildAction(rule=compute_demands_rule)


def stochd_init(m, k, j, t):
    # What it should be to match description in paper
    # if t < m.TDEC:
    #     return m.d[j]
    # if t >= m.TDEC and t < m.TDEC+5:
    #     return m.rand_d[k,j]
    # if t >= m.TDEC+5:
    #     return m.d[j]
    if t < m.TDEC + 1:
        return m.d[j]
    if t >= m.TDEC + 1 and t < m.TDEC + 1 + 4.5:
        return m.rand_d[k, j]
    if t >= m.TDEC + 1 + 4.5:
        return m.d[j]


model.stochd = pyo.Param(
    model.SCEN,
    model.DEM,
    model.TIME,
    within=pyo.PositiveReals,
    mutable=True,
    default=stochd_init,
)


# define temporal variables
def p_bounds_rule(m, k, j, t):
    return (pyo.value(m.pmin[j]), pyo.value(m.pmax[j]))


model.p = pyo.Var(
    model.SCEN, model.NODE, model.TIME, bounds=p_bounds_rule, initialize=50.0
)
model.dp = pyo.Var(
    model.SCEN, model.LINK_A, model.TIME, bounds=(0.0, 100.0), initialize=10.0
)
model.fin = pyo.Var(
    model.SCEN, model.LINK, model.TIME, bounds=(1.0, 500.0), initialize=100.0
)
model.fout = pyo.Var(
    model.SCEN, model.LINK, model.TIME, bounds=(1.0, 500.0), initialize=100.0
)


def s_bounds_rule(m, k, j, t):
    return (0.01, pyo.value(m.smax[j]))


model.s = pyo.Var(
    model.SCEN, model.SUP, model.TIME, bounds=s_bounds_rule, initialize=10.0
)
model.dem = pyo.Var(model.SCEN, model.DEM, model.TIME, initialize=100.0)
model.pow = pyo.Var(
    model.SCEN, model.LINK_A, model.TIME, bounds=(0.0, 3000.0), initialize=1000.0
)
model.slack = pyo.Var(
    model.SCEN, model.LINK, model.TIME, model.DIS, bounds=(0.0, None), initialize=10.0
)

# define spatio-temporal variables
model.px = pyo.Var(
    model.SCEN, model.LINK, model.TIME, model.DIS, bounds=(10.0, 100.0), initialize=50.0
)
model.fx = pyo.Var(
    model.SCEN, model.LINK, model.TIME, model.DIS, bounds=(1.0, 100.0), initialize=100.0
)

# define derivatives
model.dpxdt = DerivativeVar(model.px, wrt=model.TIME, initialize=0)
model.dpxdx = DerivativeVar(model.px, wrt=model.DIS, initialize=0)
model.dfxdt = DerivativeVar(model.fx, wrt=model.TIME, initialize=0)
model.dfxdx = DerivativeVar(model.fx, wrt=model.DIS, initialize=0)

# ----------- MODEL --------------


# compressor equations
def powereq_rule(m, j, i, t):
    return m.pow[j, i, t] == m.c4 * m.fin[j, i, t] * (
        ((m.p[j, m.lstartloc[i], t] + m.dp[j, i, t]) / m.p[j, m.lstartloc[i], t])
        ** m.om
        - 1.0
    )


model.powereq = pyo.Constraint(model.SCEN, model.LINK_A, model.TIME, rule=powereq_rule)

# cvar model
model.cvar_lambda = pyo.Param(within=pyo.NonNegativeReals)
model.nu = pyo.Var(initialize=100.0)
model.phi = pyo.Var(model.SCEN, bounds=(0.0, None), initialize=100.0)


def cvarcost_rule(m):
    return (1.0 / m.S) * sum((m.phi[k] / (1.0 - 0.95) + m.nu) for k in m.SCEN)


model.cvarcost = pyo.Expression(rule=cvarcost_rule)


# node balances
def nodeeq_rule(m, k, i, t):
    return (
        sum(m.fout[k, j, t] for j in m.LINK if m.lendloc[j] == i)
        + sum(m.s[k, j, t] for j in m.SUP if m.sloc[j] == i)
        - sum(m.fin[k, j, t] for j in m.LINK if m.lstartloc[j] == i)
        - sum(m.dem[k, j, t] for j in m.DEM if m.dloc[j] == i)
        == 0.0
    )


model.nodeeq = pyo.Constraint(model.SCEN, model.NODE, model.TIME, rule=nodeeq_rule)


# boundary conditions flow
def flow_start_rule(m, j, i, t):
    return m.fx[j, i, t, m.DIS.first()] == m.fin[j, i, t]


model.flow_start = pyo.Constraint(
    model.SCEN, model.LINK, model.TIME, rule=flow_start_rule
)


def flow_end_rule(m, j, i, t):
    return m.fx[j, i, t, m.DIS.last()] == m.fout[j, i, t]


model.flow_end = pyo.Constraint(model.SCEN, model.LINK, model.TIME, rule=flow_end_rule)


# First PDE for gas network model
def flow_rule(m, j, i, t, k):
    if t == m.TIME.first() or k == m.DIS.last():
        return pyo.Constraint.Skip  # Do not apply pde at initial time or final location
    return (
        m.dpxdt[j, i, t, k] / 3600 + m.c1[i] / m.llength[i] * m.dfxdx[j, i, t, k] == 0
    )


model.flow = pyo.Constraint(
    model.SCEN, model.LINK, model.TIME, model.DIS, rule=flow_rule
)


# Second PDE for gas network model
def press_rule(m, j, i, t, k):
    if t == m.TIME.first() or k == m.DIS.last():
        return pyo.Constraint.Skip  # Do not apply pde at initial time or final location
    return (
        m.dfxdt[j, i, t, k] / 3600
        == -m.c2[i] / m.llength[i] * m.dpxdx[j, i, t, k] - m.slack[j, i, t, k]
    )


model.press = pyo.Constraint(
    model.SCEN, model.LINK, model.TIME, model.DIS, rule=press_rule
)


def slackeq_rule(m, j, i, t, k):
    if t == m.TIME.last():
        return pyo.Constraint.Skip
    return (
        m.slack[j, i, t, k] * m.px[j, i, t, k]
        == m.c3[i] * m.fx[j, i, t, k] * m.fx[j, i, t, k]
    )


model.slackeq = pyo.Constraint(
    model.SCEN, model.LINK, model.TIME, model.DIS, rule=slackeq_rule
)


# boundary conditions pressure, passive links
def presspas_start_rule(m, j, i, t):
    return m.px[j, i, t, m.DIS.first()] == m.p[j, m.lstartloc[i], t]


model.presspas_start = pyo.Constraint(
    model.SCEN, model.LINK_P, model.TIME, rule=presspas_start_rule
)


def presspas_end_rule(m, j, i, t):
    return m.px[j, i, t, m.DIS.last()] == m.p[j, m.lendloc[i], t]


model.presspas_end = pyo.Constraint(
    model.SCEN, model.LINK_P, model.TIME, rule=presspas_end_rule
)


# boundary conditions pressure, active links
def pressact_start_rule(m, j, i, t):
    return m.px[j, i, t, m.DIS.first()] == m.p[j, m.lstartloc[i], t] + m.dp[j, i, t]


model.pressact_start = pyo.Constraint(
    model.SCEN, model.LINK_A, model.TIME, rule=pressact_start_rule
)


def pressact_end_rule(m, j, i, t):
    return m.px[j, i, t, m.DIS.last()] == m.p[j, m.lendloc[i], t]


model.pressact_end = pyo.Constraint(
    model.SCEN, model.LINK_A, model.TIME, rule=pressact_end_rule
)


# fix pressure at supply nodes
def suppress_rule(m, k, j, t):
    return m.p[k, m.sloc[j], t] == m.pmin[m.sloc[j]]


model.suppress = pyo.Constraint(model.SCEN, model.SUP, model.TIME, rule=suppress_rule)


# discharge pressure for compressors
def dispress_rule(m, j, i, t):
    return m.p[j, m.lstartloc[i], t] + m.dp[j, i, t] <= m.pmax[m.lstartloc[i]]


model.dispress = pyo.Constraint(
    model.SCEN, model.LINK_A, model.TIME, rule=dispress_rule
)


# ss constraints
def flow_ss_rule(m, j, i, k):
    if k == m.DIS.last():
        return pyo.Constraint.Skip
    return m.dfxdx[j, i, m.TIME.first(), k] / m.llength[i] == 0.0


model.flow_ss = pyo.Constraint(model.SCEN, model.LINK, model.DIS, rule=flow_ss_rule)


def pres_ss_rule(m, j, i, k):
    if k == m.DIS.last():
        return pyo.Constraint.Skip
    return (
        0.0
        == -m.c2[i] / m.llength[i] * m.dpxdx[j, i, m.TIME.first(), k]
        - m.slack[j, i, m.TIME.first(), k]
    )


model.pres_ss = pyo.Constraint(model.SCEN, model.LINK, model.DIS, rule=pres_ss_rule)


# non-anticipativity constraints
def nonantdq_rule(m, j, i, t):
    if j == 1:
        return pyo.Constraint.Skip
    if t >= m.TDEC + 1:
        return pyo.Constraint.Skip
    return m.dp[j, i, t] == m.dp[1, i, t]


model.nonantdq = pyo.Constraint(
    model.SCEN, model.LINK_A, model.TIME, rule=nonantdq_rule
)


def nonantde_rule(m, j, i, t):
    if j == 1:
        return pyo.Constraint.Skip
    if t >= m.TDEC + 1:
        return pyo.Constraint.Skip
    return m.dem[j, i, t] == m.dem[1, i, t]


model.nonantde = pyo.Constraint(model.SCEN, model.DEM, model.TIME, rule=nonantde_rule)
